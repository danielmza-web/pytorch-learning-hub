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
  function stageFlow(title,stages){
    const figure=el('figure',undefined,'concept-flow');figure.append(el('figcaption',title));const list=el('ol');
    stages.forEach(([name,text])=>{const item=el('li');item.append(el('b',name),el('span',text));list.append(item);});figure.append(list);return figure;
  }
  function plotCurves(canvas, strength){
    const result=curveScenario(strength), ctx=canvas.getContext('2d');if(!ctx||!result)return result;
    const w=canvas.width,h=canvas.height,left=48,top=32,right=w-15,bottom=h-34;
    ctx.clearRect(0,0,w,h);ctx.fillStyle='#101722';ctx.fillRect(0,0,w,h);ctx.font='14px system-ui';ctx.fillStyle='#d6e6f4';ctx.strokeStyle='#65798d';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
    [0,.5,1,1.5].forEach(v=>{const y=bottom-v/1.6*(bottom-top);ctx.fillText(String(v),12,y+4);});ctx.fillText('Loss',10,18);ctx.fillText('Epoch (illustrative)',w/2-45,h-6);
    [0,15,29].forEach(v=>ctx.fillText(String(v),left+v/29*(right-left)-4,bottom+16));
    for(const [name,color,values,x] of [['Training','#54cfff',result.train,55],['Validation','#ffb277',result.validation,w-95]]){
      ctx.fillStyle=color;ctx.fillText(name,x,18);ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();
      values.forEach((v,i)=>{const px=left+i/29*(right-left),py=bottom-v/1.6*(bottom-top);if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py);});ctx.stroke();
    }
    canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`${result.description} Final training loss ${fmt(result.train.at(-1))}; validation ${fmt(result.validation.at(-1))}. Synthetic curves, not a measurement.`);return result;
  }
  function svgScene(kind){
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 100 80');svg.setAttribute('role','img');svg.setAttribute('aria-label',`${kind}: the same image geometry, box and mask transformed together`);
    const shape=(tag,attrs)=>{const n=document.createElementNS(svg.namespaceURI,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));svg.append(n);return n;};
    shape('rect',{width:100,height:80,fill:'#b8c8d4'});
    const corners=[[20,15],[50,15],[50,45],[20,45]].map(([x,y])=>transformPoint(x,y,kind));
    shape('polygon',{points:corners.map(p=>p.join(',')).join(' '),fill:'#223b51'});
    const mask=[[23,18],[47,18],[39,40],[23,40]].map(([x,y])=>transformPoint(x,y,kind));
    shape('polygon',{points:mask.map(p=>p.join(',')).join(' '),fill:'#ee7b29',opacity:.6});
    const b=transformBox([20,15,30,30],kind);if(b)shape('rect',{x:b[0],y:b[1],width:b[2],height:b[3],fill:'none',stroke:'#006f83','stroke-width':2,'stroke-dasharray':'3 2'});
    return svg;
  }
  let serial=0;
  function initialize(root){
    if(root.dataset.learningReady)return;root.dataset.learningReady='true';
    const kind=root.dataset.learningLab, controls=el('div',undefined,'learning-controls'),result=el('div',undefined,'learning-result'),status=el('output');status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    const defaults=[], values={};let update;
    function control(name,label,type,initial,options){
      const wrap=el('label'),id=`learning-${++serial}`;wrap.htmlFor=id;wrap.append(el('span',label));let input;
      if(type==='select'){input=el('select');options.forEach(([v,text])=>{const o=el('option',text);o.value=v;input.append(o);});}
      else{input=el('input');input.type=type;for(const [k,v] of Object.entries(options||{}))input.setAttribute(k,String(v));}
      input.id=id;input.dataset.control=name;
      if(type==='checkbox')input.checked=initial;else input.value=String(initial);
      wrap.append(input);controls.append(wrap);values[name]=input;defaults.push(()=>{if(type==='checkbox')input.checked=initial;else input.value=String(initial);});
      input.addEventListener(type==='select'||type==='checkbox'?'change':'input',()=>update());return input;
    }
    const number=name=>values[name].value.trim()?Number(values[name].value):NaN;
    const choice=name=>values[name].value;
    function button(label,action){const b=el('button',label);b.type='button';b.addEventListener('click',action);controls.append(b);return b;}
    root.replaceChildren();root.classList.add('learning-lab');root.append(controls,result,status);
    function show(nodes,text){result.replaceChildren(...nodes);status.textContent=text;}
    function invalid(message){show([],message);}
    let phase=0;
    if(kind==='training'){
      control('lr','Learning rate','number',.1,{min:0,max:1,step:.05});
      const previous=button('Previous step',()=>{phase=Math.max(0,phase-1);update();}),next=button('Next step',()=>{phase=Math.min(4,phase+1);update();});
      update=()=>{const a=training(number('lr'));if(!a)return invalid('Use a learning rate from 0 to 1.');previous.disabled=phase===0;next.disabled=phase===4;
        const names=['Clear gradients','Forward','Loss','Backward','Update'];
        const stages=stageFlow('One SGD update · two samples',names.map((n,i)=>[`${i+1}. ${n}`,i===phase?'Current step':i<phase?'Already completed':'Still to come']));
        const nodes=[stages,matrix(phase===4?'Updated weights':'Weights: unchanged until step 5',phase===4?a.updated:a.weights)];
        if(phase>=1)nodes.push(matrix('Logits = x @ w.T',a.logits));if(phase>=2)nodes.push(el('p',`Mean cross-entropy: ${fmt(a.loss)} (computed before this update).`));if(phase>=3)nodes.push(matrix('Gradients: backward computes these',a.gradients));
        show(nodes,`Step ${phase+1}: ${names[phase]}. ${phase===4?'Weights now change by −learning rate × gradient.':'Weights remain fixed.'} Synthetic arithmetic; no model training runs in this page.`);
      };
    }else if(kind==='tensor-ops'){
      control('op','Operation','select','reshape',[['reshape','reshape (3, 2)'],['transpose','transpose / permute'],['cat','cat along rows'],['stack','stack on a new axis']]);
      update=()=>{const op=choice('op'),a=[[1,2,3],[4,5,6]],b=[[7,8,9],[10,11,12]],out=op==='reshape'?[[1,2],[3,4],[5,6]]:op==='transpose'?[[1,4],[2,5],[3,6]]:op==='cat'?[...a,...b]:null;
        const nodes=[matrix('A · shape [2,3]',a)];if(['cat','stack'].includes(op))nodes.push(matrix('B · shape [2,3]',b));if(out)nodes.push(matrix('Result',out));else nodes.push(matrix('Result [0] = A',a),matrix('Result [1] = B',b));
        show(nodes,op==='reshape'?'[2,3] → [3,2]: consecutive values are regrouped. Axes are not semantically swapped.':op==='transpose'?'[2,3] → [3,2]: row and column axes are exchanged. Same shape as reshape, different values.':op==='cat'?'Two [2,3] inputs → [4,3]: join the existing row axis.':'Two [2,3] inputs → [2,2,3]: create a new axis for the two inputs.');
      };
    }else if(kind==='convolution'){
      control('kernel','Filter','select','edge',[['edge','Vertical edge'],['horizontal','Horizontal edge'],['sharpen','Sharpen'],['blur','Mean blur']]);
      control('stride','Stride','select','1',[['1','1'],['2','2']]);control('pad','Zero padding','select','0',[['0','0'],['1','1']]);
      const pos=control('position','Output cell to inspect','range',0,{min:0,max:15,step:1});
      update=()=>{const stride=number('stride'),pad=number('pad'),kernel=KERNELS[choice('kernel')],out=convolution(INPUT,kernel,stride,pad);pos.max=out.length**2-1;const index=Math.min(number('position'),Number(pos.max));pos.value=index;
        const r=Math.floor(index/out.length),c=index%out.length,top=r*stride-pad,left=c*stride-pad;
        const patch=kernel.map((row,kr)=>row.map((_,kc)=>INPUT[top+kr]?.[left+kc]??0));
        const products=patch.map((row,kr)=>row.map((v,kc)=>v*kernel[kr][kc]));
        show([matrix('Input · 6×6',INPUT,(y,x)=>y>=top&&y<top+3&&x>=left&&x<left+3),matrix('3×3 filter',kernel),matrix('Selected window · outside input = 0',patch),matrix('Elementwise products',products),matrix(`Output · ${out.length}×${out.length}`,out,(y,x)=>y===r&&x===c)],`Output [${r},${c}] = sum of nine products = ${fmt(out[r][c])}. Stride ${stride}; padding ${pad}. Cross-correlation, as used by Conv2d; no kernel reversal.`);
      };
    }else if(kind==='threshold'){
      control('threshold','Flag positive when score ≥ threshold','range',.5,{min:0,max:1,step:.1});
      update=()=>{const a=threshold(number('threshold'));if(!a)return invalid('Use a threshold from 0 to 1.');
        show([table(`Threshold ${fmt(number('threshold'))} · fixed synthetic scores`,['Sample','Score','Truth','Prediction'],a.scores.map((s,i)=>[i+1,s,a.truth[i],a.predicted[i]])),table('Confusion matrix · positive = defect',['Actual / predicted','Defect','Good'],[['Defect',a.tp,a.fn],['Good',a.fp,a.tn]])],`TP ${a.tp}, FP ${a.fp}, FN ${a.fn}, TN ${a.tn}. Precision ${a.precision===null?'N/A (no alarms)':fmt(a.precision)}; recall ${fmt(a.recall)}; accuracy ${fmt(a.accuracy)}. Changing the threshold changes decisions, not the scores or trained weights.`);
      };
    }else if(kind==='accumulation'){
      control('size','Microbatch samples','number',16,{min:1,max:1000000});control('steps','Microbatches per update','number',3,{min:1,max:1000000});control('total','Epoch samples','number',133,{min:1,max:1000000000});
      update=()=>{const size=number('size'),steps=number('steps'),total=number('total');const a=accumulation(size,steps,total);if(!a)return invalid('Use positive whole numbers with a safe effective-batch product.');
        const description=g=>`${g.full} × ${size}${g.remainder?` + ${g.remainder}`:''} samples in ${g.count} microbatches`;
        show([stageFlow('First optimizer update',[['Clear','zero_grad before the group'],['Accumulate',description(a.first)],['Normalize',`Divide summed unweighted per-sample loss gradients by ${Math.min(total,a.effective)} actual samples`],['Update','step once; then clear for the next group']]),table('Full versus final group',['Group','Samples','Microbatches'],[['Full capacity',a.effective,steps],['Final update',a.last,a.final.count]])],`${a.updates} optimizer updates. Final group: ${description(a.final)}; divide by ${a.last}, not ${a.effective}. BatchNorm still sees each microbatch; class-weighted losses need their own denominator. No speed prediction.`);
      };
    }else if(kind==='paired-transform'){
      control('geometry','Shared geometry','select','flip',[['identity','Unchanged'],['flip','Horizontal flip'],['crop','Crop [20,10,60,60] and resize']]);
      update=()=>{const k=choice('geometry'),before=el('figure'),after=el('figure');before.append(el('figcaption','Before · pixels + box + mask'),svgScene('identity'));after.append(el('figcaption','After · one shared transform'),svgScene(k));
        show([before,after],`Box [x,y,width,height]: [20,15,30,30] → [${transformBox([20,15,30,30],k).map(fmt).join(', ')}]. Blue dashed box and orange mask move with the dark object. Integer class masks are not image-normalized. Original geometry, not model predictions.`);
      };
    }else if(kind==='transfer'){
      control('strategy','Training strategy','select','head',[['head','Head only'],['partial','Last block + head'],['full','All blocks']]);control('strict','Frozen backbone stays in eval mode','checkbox',true);
      update=()=>{const strategy=choice('strategy'),trainable=[strategy==='full',strategy!=='head',true],counts=[20,20,10],sum=counts.reduce((s,n,i)=>s+(trainable[i]?n:0),0);
        show([table('Toy parameter ownership · two 4→4 blocks and a 4→2 head',['Component','Stored values','Gradients / optimizer','BN / Dropout mode policy'],['Early block','Late block','New head'].map((n,i)=>[n,counts[i],trainable[i]?'Included':'Frozen',trainable[i]||!values.strict.checked?'train: BN may update; Dropout active':'eval: BN fixed; Dropout off']))],`${sum} of 50 toy values trainable; all 50 remain stored and all blocks still run forward. ${!values.strict.checked?'Frozen weights alone do not freeze BatchNorm statistics or disable Dropout.':'This deliberate eval policy keeps frozen-block behavior stable.'} BN/Dropout are policy examples, excluded from this toy parameter count.`);
      };
    }else if(kind==='embedding'){
      control('token','Toy token ID','select','2',[['2','2 · clear'],['3','3 · image'],['4','4 · sharp'],['0','0 · PAD']]);control('position','Position','select','0',[['0','0'],['1','1'],['2','2']]);
      update=()=>{const id=number('token'),p=number('position'),vector=EMBEDDINGS[id],signal=[p/10,-p/10];
        show([table('Fixed toy table · coordinates are not meanings',['ID','Vector'],Object.entries(EMBEDDINGS).map(([k,v])=>[k,`[${v}]`])),matrix('Lookup vector',[vector]),matrix('Toy positional signal',[signal]),matrix('Vector + position (before contextual mixing)',[vector.map((v,i)=>v+signal[i])])],`ID ${id} always looks up [${vector}]. Position ${p} changes a separate positional signal, not the table row. Context requires a trained mixing model; this addition alone supplies no semantics. PAD still needs attention and pooling masks.`);
      };
    }else if(kind==='pooling'){
      control('sequence','Sequence','select','2,3,0',[['2,3,0','clear image PAD'],['3,2,0','image clear PAD'],['2,3,4','clear image sharp'],['0,0,0','PAD PAD PAD']]);control('mode','Pooling','select','mean',[['mean','Mean'],['sum','Sum'],['max','Max']]);control('masked','Exclude PAD','checkbox',true);
      update=()=>{const ids=choice('sequence').split(',').map(Number),mode=choice('mode'),a=pooling(ids,mode,values.masked.checked),correct=pooling(ids,mode,true);
        show([matrix('Token vectors',ids.map(id=>EMBEDDINGS[id])),matrix(`${mode} result`,[a.vector]),table('Padding check',['Calculation','Included rows','Vector'],[['Selected rule',a.count,`[${a.vector}]`],['PAD excluded',correct.count,`[${correct.vector}]`]])],a.empty?'All PAD: this toy explicitly returns [0,0]. A real system may reject empty text or use UNK.':`${a.count} rows included. ${mode==='mean'?'Mean divides by this count.':mode==='sum'?'Sum grows with additional tokens.':'Each coordinate takes its largest included value; zero padding can win over negative real values.'} Reversing tokens preserves all three pools: order is lost. Toy values, not trained semantics.`);
      };
    }else if(kind==='connections'){
      control('connection','Connection','select','residual',[['residual','Residual addition'],['dense','Dense concatenation'],['mobile','Depthwise + pointwise']]);control('channels','Input channels','number',16,{min:1,max:128});control('added','New / output channels','number',32,{min:1,max:128});
      update=()=>{const a=connection(choice('connection'),number('channels'),number('added'));if(!a)return invalid('Use whole channel counts from 1 to 128.');
        show([stageFlow('Follow the channels',[['Input',`[N,${number('channels')},H,W]`],['Mechanism',a.description],['Output',`${a.channels} channels; same H/W in this illustration`]])],a.weights===null?(choice('connection')==='residual'?'Addition requires aligned shapes. The correction uses the input channel count; a different-width main path needs an explicit projection.':'Concatenation retains old channels and appends new ones; it is not addition.'):`3×3 dense: ${a.denseWeights} weights; depthwise + 1×1: ${a.weights}. Bias excluded; equal spatial output assumed. This count is not measured latency.`);
      };
    }else if(kind==='attention'){
      control('kind','Attention access','select','causal',[['causal','Causal'],['encoder','Bidirectional']]);control('padding','Position 2 is PAD','checkbox',false);control('query','Query row','select','1',[['0','0'],['1','1'],['2','2']]);
      update=()=>{const a=attention(choice('kind'),values.padding.checked),q=number('query');
        show([matrix('Q = K · fixed [3,2] vectors',a.q,(r)=>r===q),matrix('QKᵀ / √2 · [3,3] scores',a.scores,(r)=>r===q),matrix('Allowed pairs · ✓ permitted, × blocked',a.allowed.map(row=>row.map(v=>v?'✓':'×')),(r)=>r===q),matrix('Softmax over permitted keys',a.weights,(r)=>r===q),matrix('Value vectors',a.values),matrix('Weighted value output',a.output,(r)=>r===q)],`Query ${q}: weights [${a.weights[q].map(fmt)}], sum ${fmt(a.weights[q].reduce((s,v)=>s+v,0))}; output [${a.output[q].map(fmt)}]. ${values.padding.checked?'PAD key 2 is excluded, but its query output is not automatically zero.':''} Synthetic calculations; not learned attention or language quality.`);
      };
    }else if(kind==='lifecycle'){
      control('artifact','Saved artifact','select','weights',[['weights','Full model weights + contract'],['checkpoint','Training checkpoint'],['encoder','Encoder only']]);
      const previous=button('Previous step',()=>{phase=Math.max(0,phase-1);update();}),next=button('Next step',()=>{phase=Math.min(5,phase+1);update();});
      update=()=>{previous.disabled=phase===0;next.disabled=phase===5;const type=choice('artifact'),contents=type==='checkpoint'?'Model + optimizer + run state + input contract':type==='encoder'?'Encoder state + its feature/input contract':'Model state + architecture/input contract';
        const names=['Construct','Train','Save','Close process','Reconstruct','Restore'];const descriptions=['Architecture creates live initialized tensors.','An update changes values in RAM.',`Write ${contents.toLowerCase()} to disk.`,'Live tensors disappear; the saved artifact survives.','Compatible architecture code creates a new live object.',type==='checkpoint'?'Restore model and optimizer/run state to resume deliberately.':type==='encoder'?'Restore the encoder; the next head is separate.':'Restore model tensors; set eval and disable gradients for inference.'];
        show([stageFlow('Memory → storage → a new process',names.map((n,i)=>[`${i+1}. ${n}`,i===phase?descriptions[i]:i<phase?'Completed':'Next'])),table('What remains at this step',['Place','Contents'],[['RAM',phase===3?'Nothing from the closed process':phase<4?'Original live model':'New compatible model'],['Disk',phase>=2?contents:'No saved artifact yet']])],`Step ${phase+1}: ${descriptions[phase]} ${phase===5&&type==='weights'?'Weights alone do not restore Adam moments.':''} Conceptual lifecycle; this page does not save files or train a model.`);
      };
    }else if(kind==='compression'){
      control('representation','Representation','select','zeros',[['zeros','Mask half the FP32 entries'],['smaller','Actually smaller FP32 layer'],['int8','Hypothetical INT8 payload']]);control('value','Value to quantize','number',1.3,{min:-4,max:4,step:.05});control('scale','Toy quantization scale','select','.25',[['.25','0.25'],['.5','0.5']]);
      update=()=>{const value=number('value');if(!Number.isFinite(value)||value< -4||value>4)return invalid('Use a finite value from −4 to 4.');const k=choice('representation'),rows=k==='smaller'?4:8,bits=k==='int8'?8:32,grid=Array.from({length:rows},(_,r)=>Array.from({length:4},(_,c)=>k==='zeros'&&(r+c)%2===0?0:1)),a=quantize(value,number('scale'));
        show([matrix(`${rows}×4 stored slots · ${rows*4*bits/8} payload bytes`,grid),table('One quantization calculation · toy signed 4-bit range −8…7',['Stage','Value'],[['Original',value],['Round(value / scale), ties to even',a.raw],['Clip to supported integer range',a.integer],['Reconstruct integer × scale',a.restored],['Error (restored − original)',fmt(a.error)]])],`${k==='zeros'?'16 zeros still occupy 32 dense FP32 slots (128 bytes).':k==='smaller'?'16 FP32 slots (64 bytes); connected layers must also be adapted.':'32 hypothetical INT8 slots (32 bytes); metadata and supported kernels add costs.'} ${a.clipped?'This toy value clips at the range boundary.':'This toy illustrates rounding.'} The numeric demo uses a small 4-bit range for visibility; the payload comparison uses the stated FP32/INT8 format. Neither predicts speed or accuracy.`);
      };
    }else{return invalid('This demonstration is unavailable.');}
    button('Reset example',()=>{defaults.forEach(reset=>reset());phase=0;update();});update();
  }
  function init(){
    document.querySelectorAll('[data-learning-lab]').forEach(initialize);
    document.querySelectorAll('[data-curve-lab]').forEach(root=>{
      if(root.dataset.curveReady)return;root.dataset.curveReady='true';
      const input=root.querySelector('[data-regularization]'),canvas=root.querySelector('canvas'),output=root.querySelector('[data-curve-output]');
      const update=()=>{
        const width=Math.round(canvas.getBoundingClientRect().width);
        if(width>0&&canvas.width!==width){canvas.width=width;canvas.height=240;}
        const a=plotCurves(canvas,Number(input.value)/100);
        if(a)output.textContent=`${a.description} Final train ${fmt(a.train.at(-1))}; validation ${fmt(a.validation.at(-1))}. Three synthetic regimes, not a measured regularization response.`;
      };
      input.addEventListener('input',update);update();
      const reset=el('button','Reset curves');reset.type='button';
      reset.addEventListener('click',()=>{input.value=input.defaultValue;update();});root.insertBefore(reset,canvas);
      if(typeof ResizeObserver!=='undefined')new ResizeObserver(update).observe(canvas);
    });
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
