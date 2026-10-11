/* Original small calculations, shared by two independently hosted static sites. */
(function () {
  'use strict';
  const INPUT = [[0,0,0,1,1,1],[0,0,0,1,1,1],[0,0,.1,.9,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1],[0,0,0,1,1,1]];
  const KERNELS = {edge:[[-1,0,1],[-1,0,1],[-1,0,1]], horizontal:[[-1,-1,-1],[0,0,0],[1,1,1]], sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]], blur:Array.from({length:3},()=>[1/9,1/9,1/9])};
  const fmt = n => Number.isFinite(n) ? String(Number(n.toFixed(4))) : 'N/A';
  function softmax(values) {
    const maximum = Math.max(...values), exp = values.map(v=>Math.exp(v-maximum)), sum = exp.reduce((a,b)=>a+b,0);
    return exp.map(v=>v/sum);
  }
  function training(lr) {
    if (!Number.isFinite(lr) || lr < 0 || lr > 1) return null;
    const x=[[1,2],[3,4]], weights=[[-4,3],[2,-1]], targets=[0,1];
    const logits=x.map(row=>weights.map(w=>row.reduce((s,v,i)=>s+v*w[i],0)));
    const probabilities=logits.map(softmax);
    const loss=-probabilities.reduce((sum,p,i)=>sum+Math.log(p[targets[i]]),0)/2;
    const gradients=weights.map((w,k)=>w.map((_,f)=>x.reduce((sum,row,i)=>sum+(probabilities[i][k]-(targets[i]===k?1:0))*row[f],0)/2));
    return {weights,logits,probabilities,loss,gradients,updated:weights.map((w,k)=>w.map((v,f)=>v-lr*gradients[k][f]))};
  }
  function convolution(input, kernel, stride=1, padding=0) {
    if (!Number.isInteger(stride) || stride<1 || stride>2 || !Number.isInteger(padding) || padding<0 || padding>1) return null;
    const size=Math.floor((input.length+2*padding-3)/stride)+1;
    if (size<1) return null;
    return Array.from({length:size},(_,r)=>Array.from({length:size},(_,c)=>kernel.reduce((sum,row,kr)=>sum+row.reduce((s,k,kc)=>s+k*(input[r*stride+kr-padding]?.[c*stride+kc-padding] ?? 0),0),0)));
  }
  function threshold(t) {
    if (!Number.isFinite(t) || t<0 || t>1) return null;
    const scores=[.9,.6,.4,.1], truth=[1,0,1,0], predicted=scores.map(p=>Number(p>=t));
    const counts={tp:0,fp:0,fn:0,tn:0};
    truth.forEach((y,i)=>counts[y ? (predicted[i]?'tp':'fn') : (predicted[i]?'fp':'tn')]++);
    const {tp,fp,fn,tn}=counts;
    return {scores,truth,predicted,...counts,precision:tp+fp?tp/(tp+fp):null,recall:tp/(tp+fn),accuracy:(tp+tn)/4};
  }
  function accumulation(size, steps, total) {
    if (![size,steps,total].every(v=>Number.isSafeInteger(v)&&v>0) || !Number.isSafeInteger(size*steps)) return null;
    const effective=size*steps, updates=Math.ceil(total/effective), last=total%effective || Math.min(total,effective);
    const group=n=>{const count=Math.ceil(n/size);return {count,full:Math.floor(n/size),remainder:n%size};};
    return {effective,updates,last,first:group(Math.min(total,effective)),final:group(last)};
  }
  const EMBEDDINGS={0:[0,0],2:[2,-2],3:[4,-4],4:[1,-1]};
  function pooling(ids, mode, masked=true) {
    if (!ids.length || !['mean','sum','max'].includes(mode) || ids.some(id=>!Object.hasOwn(EMBEDDINGS,id))) return null;
    const rows=ids.filter(id=>!masked || id!==0).map(id=>EMBEDDINGS[id]);
    if (!rows.length) return {vector:[0,0],count:0,empty:true};
    const vector=[0,1].map(d=>mode==='max'?Math.max(...rows.map(row=>row[d])):rows.reduce((sum,row)=>sum+row[d],0)/(mode==='mean'?rows.length:1));
    return {vector,count:rows.length,empty:false};
  }
  function connection(kind, channels=16, added=32) {
    if (![channels,added].every(v=>Number.isInteger(v)&&v>0&&v<=128)) return null;
    if (kind==='residual') return {channels,weights:null,description:`[N,${channels},H,W] + [N,${channels},H,W] → [N,${channels},H,W]`};
    if (kind==='dense') return {channels:channels+added,weights:null,description:`cat([N,${channels},H,W], [N,${added},H,W], dim=1) → [N,${channels+added},H,W]`};
    if (kind==='mobile') return {channels:added,weights:9*channels+channels*added,denseWeights:9*channels*added,description:`Depthwise: ${channels} spatial filters → pointwise: ${channels} to ${added} channels`};
    return null;
  }
  function attention(kind, padding=false) {
    if (!['encoder','causal'].includes(kind)) return null;
    const q=[[1,0],[0,1],[1,1]], k=q, values=[[2,0],[0,4],[3,3]];
    const scores=q.map(row=>k.map(key=>row.reduce((sum,v,d)=>sum+v*key[d],0)/Math.sqrt(2)));
    const allowed=q.map((_,r)=>k.map((_,c)=>(kind==='encoder'||c<=r)&&(!padding||c!==2)));
    const weights=scores.map((row,r)=>softmax(row.map((v,c)=>allowed[r][c]?v:-Infinity)));
    return {q,k,scores,allowed,weights,output:weights.map(row=>[0,1].map(d=>row.reduce((sum,w,c)=>sum+w*values[c][d],0))),values};
  }
  function quantize(value,scale=.25) {
    if (!Number.isFinite(value)||!Number.isFinite(scale)||scale<=0) return null;
    const scaled=value/scale, lower=Math.floor(scaled);
    const raw=scaled-lower===.5?(lower%2===0?lower:lower+1):Math.round(scaled);
    const integer=Math.max(-8,Math.min(7,raw)), restored=integer*scale;
    return {raw,integer,restored,error:restored-value,clipped:integer!==raw};
  }
  function curveScenario(strength) {
    if (!Number.isFinite(strength)||strength<0||strength>1) return null;
    // Select three illustrative regimes; this is not a fitted regularization law.
    const underfit=strength>.78?.6+(strength-.78)*.3:0;
    const overfit=strength<.25?.9+(.25-strength)*.4:0;
    const train=[],validation=[];
    for(let i=0;i<30;i++){const t=i/29;train.push(.75*Math.exp(-4*t)+.08+underfit);validation.push(.72*Math.exp(-3.4*t)+.15+underfit+overfit*Math.max(0,t-.4)**2);}
    return {train,validation,description:strength<.25?'Overfitting: training falls while validation turns upward.':strength>.78?'Underfitting: both losses remain high.':'Balanced illustration: both losses fall, with a modest gap.'};
  }
  function transformPoint(x,y,kind) {
    if(kind==='flip') return [100-x,y];
    if(kind==='crop') return [(x-20)*100/60,(y-10)*80/60];
    return [x,y];
  }
  function transformBox(box,kind) {
    const a=transformPoint(box[0],box[1],kind), b=transformPoint(box[0]+box[2],box[1]+box[3],kind);
    const x=Math.max(0,Math.min(...[a[0],b[0]])), y=Math.max(0,Math.min(a[1],b[1]));
    const right=Math.min(100,Math.max(a[0],b[0])),bottom=Math.min(80,Math.max(a[1],b[1]));
    return right>x&&bottom>y?[x,y,right-x,bottom-y]:null;
  }
  const api={training,convolution,threshold,accumulation,pooling,connection,attention,quantize,curveScenario,transformPoint,transformBox,INPUT,KERNELS,EMBEDDINGS};
  if(typeof module!=='undefined') module.exports=api;
  if(typeof document==='undefined') return;
  const el=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
  function table(title,headings,rows){
    const t=el('table',undefined,'learning-table');t.append(el('caption',title));
    const head=el('thead'),h=el('tr');headings.forEach(text=>{const c=el('th',text);c.scope='col';h.append(c);});head.append(h);t.append(head);
    const body=el('tbody');rows.forEach(row=>{const r=el('tr');row.forEach((text,i)=>{const c=el(i===0?'th':'td',String(text));if(!i)c.scope='row';r.append(c);});body.append(r);});t.append(body);return t;
  }
  function matrix(title,values,highlight){
    const figure=el('figure',undefined,'learning-matrix');figure.tabIndex=0;
    figure.setAttribute('aria-label',`${title}. Use left and right arrows if this matrix overflows.`);figure.append(el('figcaption',title));
    const grid=el('div',undefined,'learning-cells');grid.style.setProperty('--columns',values[0].length);
    values.forEach((row,r)=>row.forEach((v,c)=>{const cell=el('span',typeof v==='number'?fmt(v):v);if(highlight?.(r,c))cell.className='selected-cell';grid.append(cell);}));figure.append(grid);return figure;
  }
  let serial=0;
  function initialize(root){
    if(root.dataset.learningReady)return;root.dataset.learningReady='true';
    const kind=root.dataset.learningLab,controls=el('div',undefined,'learning-controls'),result=el('div',undefined,'learning-result'),status=el('output');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    const advanced=el('details',undefined,'learning-exploration'),advancedControls=el('div',undefined,'learning-controls'),advancedResult=el('div',undefined,'learning-result');advanced.append(el('summary','More options and calculation details'),advancedControls,advancedResult);
    const defaults=[],values={};let update;
    function control(name,label,type,initial,options,parent=controls){
      const wrap=el('label'),id=`learning-${++serial}`;wrap.htmlFor=id;wrap.append(el('span',label));let input;
      if(type==='select'){input=el('select');options.forEach(([v,text])=>{const o=el('option',text);o.value=v;input.append(o);});}
      else{input=el('input');input.type=type;for(const [k,v]of Object.entries(options||{}))input.setAttribute(k,String(v));}
      input.id=id;input.dataset.control=name;
      if(type==='checkbox')input.checked=initial;else input.value=String(initial);
      wrap.append(input);parent.append(wrap);values[name]=input;defaults.push(()=>{if(type==='checkbox')input.checked=initial;else input.value=String(initial);});
      input.addEventListener(type==='select'?'change':'input',()=>update());return input;
    }
    const number=name=>values[name].value.trim()?Number(values[name].value):NaN,choice=name=>values[name].value;
    const show=(nodes,text,extra=[])=>{result.replaceChildren(...nodes);advancedResult.replaceChildren(...extra);status.textContent=text;};
    const invalid=text=>show([],text);
    root.replaceChildren(controls,result,status);root.classList.add('learning-lab');
    if(kind==='convolution'){
      const pos=control('position','Output cell to inspect','range',1,{min:0,max:15,step:1});
      control('kernel','Filter','select','edge',[['edge','Vertical edge'],['horizontal','Horizontal edge'],['sharpen','Sharpen'],['blur','Mean blur']],advancedControls);
      control('stride','Stride','select','1',[['1','1'],['2','2']],advancedControls);control('pad','Zero padding','select','0',[['0','0'],['1','1']],advancedControls);
      root.append(advanced);
      update=()=>{const stride=number('stride'),pad=number('pad'),kernel=KERNELS[choice('kernel')],out=convolution(INPUT,kernel,stride,pad);pos.max=out.length**2-1;const index=Math.min(number('position'),Number(pos.max));pos.value=index;
        const r=Math.floor(index/out.length),c=index%out.length,top=r*stride-pad,left=c*stride-pad;
        const patch=kernel.map((row,kr)=>row.map((_,kc)=>INPUT[top+kr]?.[left+kc]??0)),products=patch.map((row,kr)=>row.map((v,kc)=>v*kernel[kr][kc]));
        show([matrix('1 · Input · highlighted window',INPUT,(y,x)=>y>=top&&y<top+3&&x>=left&&x<left+3),matrix('2 · Multiply by this filter',kernel),matrix(`3 · Output · ${out.length}×${out.length}`,out,(y,x)=>y===r&&x===c)],`Cell [${r},${c}]: sum of nine products = ${fmt(out[r][c])}. ${choice('kernel')} filter; stride ${stride}; zero padding ${pad}. Cross-correlation, no kernel reversal.`,[matrix('Selected 3×3 window · outside input = 0',patch),matrix('Nine elementwise products',products)]);
      };
    }else if(kind==='threshold'){
      control('threshold','Flag positive when score ≥ threshold','range',.5,{min:0,max:1,step:.1});
      advanced.firstChild.textContent='See confusion counts';root.append(advanced);
      update=()=>{const a=threshold(number('threshold'));if(!a)return invalid('Use a threshold from 0 to 1.');
        show([table(`Threshold ${fmt(number('threshold'))} · fixed scores and true labels`,['Sample','Score','Truth','Prediction'],a.scores.map((s,i)=>[i+1,s,a.truth[i]?'Defect':'Good',a.predicted[i]?'Defect':'Good']))],`Precision ${a.precision===null?'N/A (no alarms)':fmt(a.precision)}; recall ${fmt(a.recall)}. Decisions change; scores and trained weights stay fixed.`,[table('Truth rows · prediction columns',['Actual / predicted','Defect','Good'],[['Defect',a.tp,a.fn],['Good',a.fp,a.tn]]),el('p',`TP ${a.tp}, FP ${a.fp}, FN ${a.fn}, TN ${a.tn}; accuracy ${fmt(a.accuracy)}.`)]);
      };
    }else if(kind==='accumulation'){
      control('size','Microbatch samples','number',16,{min:1,max:1000000});control('steps','Microbatches per update','number',3,{min:1,max:1000000});control('total','Epoch samples','number',133,{min:1,max:1000000000});
      update=()=>{const a=accumulation(number('size'),number('steps'),number('total'));if(!a)return invalid('Use positive whole numbers with a safe effective-batch product.');
        show([table('Samples per optimizer update',['Group','Actual samples','Microbatches'],[['Full capacity',a.effective,number('steps')],['Final update',a.last,a.final.count]])],`${a.updates} updates. Final group: ${a.final.full} × ${number('size')}${a.final.remainder?` + ${a.final.remainder}`:''} = ${a.last}; ${a.last===a.effective?`this complete group uses divisor ${a.last}`:`divide by ${a.last}, not automatically by ${a.effective}`}. Unweighted per-sample loss only; BatchNorm still sees each microbatch.`);
      };
    }else if(kind==='attention'){
      control('kind','Attention access','select','causal',[['causal','Causal'],['encoder','Bidirectional']]);
      control('padding','Position 2 is PAD','checkbox',false,{},advancedControls);control('query','Query row','select','1',[['0','0'],['1','1'],['2','2']],advancedControls);root.append(advanced);
      update=()=>{const a=attention(choice('kind'),values.padding.checked),q=number('query');
        show([matrix(`Query ${q} · weights over keys 0, 1, 2`,[a.weights[q]]),matrix(`Query ${q} · weighted value output`,[a.output[q]])],`Query ${q}; ${choice('kind')}; key 2 ${values.padding.checked?'excluded as PAD':'is not PAD'}. Weights sum to ${fmt(a.weights[q].reduce((s,v)=>s+v,0))}. ${values.padding.checked?'Masking a PAD key does not automatically zero its query output.':''} Fixed synthetic vectors, not learned attention.`,[matrix('Q = K · [3,2]',a.q,(r)=>r===q),matrix('QKᵀ / √2 scores',a.scores,(r)=>r===q),matrix('Allowed pairs · ✓ allowed, × blocked',a.allowed.map(row=>row.map(v=>v?'✓':'×')),(r)=>r===q),matrix('Softmax · all rows',a.weights,(r)=>r===q),matrix('Value vectors',a.values),matrix('Outputs · all rows',a.output,(r)=>r===q)]);
      };
    }else if(kind==='quantization'){
      control('value','Original value','number',1.3,{min:-4,max:4,step:.05});
      update=()=>{const v=number('value');if(!Number.isFinite(v)||v< -4||v>4)return invalid('Use a finite value from −4 to 4.');const a=quantize(v,.25);
        show([table('Fixed scale 0.25 · toy signed 4-bit range −8…7',['Stage','Value'],[['Original',v],['Round(value / 0.25), ties to even',a.raw],['Clip to −8…7',a.integer],['Reconstruct integer × 0.25',a.restored],['Error (restored − original)',fmt(a.error)]])],a.clipped?'Clipped at the range boundary. This is a 4-bit arithmetic toy, separate from the INT8 payload comparison.':'Rounding changes the reconstructed value. This is a 4-bit arithmetic toy, not a quality or speed measurement.');
      };
    }else{return invalid('This demonstration is unavailable.');}
    const reset=el('button','Reset example');reset.type='button';reset.addEventListener('click',()=>{defaults.forEach(reset=>reset());update();});controls.append(reset);update();
  }
  function init(){document.querySelectorAll('[data-learning-lab]').forEach(initialize);}
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
