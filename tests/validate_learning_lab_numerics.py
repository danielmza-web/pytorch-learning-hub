"""Independently compare browser arithmetic with CPU PyTorch operations.

No training downloads, maintained-image writes or generated golden fixtures.
"""
from pathlib import Path
import importlib.util
import json
import subprocess
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CODE = """
const a=require('./docs/assets/javascripts/learning-labs.js');
process.stdout.write(JSON.stringify({
  training:[0,.1,1].map(x=>a.training(x)),
  convolution:Object.values(a.KERNELS).flatMap(k=>[1,2].flatMap(s=>[0,1].map(p=>a.convolution(a.INPUT,k,s,p)))),
  attention:['encoder','causal'].flatMap(k=>[false,true].map(p=>a.attention(k,p))),
  pooling:['mean','sum','max'].flatMap(m=>[true,false].map(mask=>a.pooling([2,3,0],m,mask))),
  quantization:[-4,-.625,-.375,0,.375,.625,1.3,4].flatMap(v=>[.25,.5].map(s=>a.quantize(v,s))),
  input:a.INPUT,kernels:a.KERNELS,
  geometry:['identity','flip','crop'].map(k=>a.transformBox([20,15,30,30],k))
}));
"""
data = json.loads(subprocess.check_output(['node','-e',CODE],cwd=ROOT,text=True))
def compare(actual,expected):
    torch.testing.assert_close(torch.tensor(actual,dtype=torch.float64),expected.to(torch.float64),rtol=1e-10,atol=1e-10)

for rate,result in zip([0,.1,1],data['training']):
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    w=torch.tensor([[-4.,3.],[2.,-1.]],dtype=torch.float64,requires_grad=True)
    logits=F.linear(x,w); loss=F.cross_entropy(logits,torch.tensor([0,1])); loss.backward()
    compare(result['logits'],logits);compare(result['probabilities'],logits.softmax(-1))
    compare(result['loss'],loss);compare(result['gradients'],w.grad)
    compare(result['updated'],w.detach()-rate*w.grad)

image=torch.tensor(data['input'],dtype=torch.float64)[None,None]
cases=iter(data['convolution'])
for kernel in data['kernels'].values():
    for stride in [1,2]:
        for padding in [0,1]:
            expected=F.conv2d(image,torch.tensor(kernel,dtype=torch.float64)[None,None],stride=stride,padding=padding)[0,0]
            compare(next(cases),expected)

q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
v=torch.tensor([[2.,0.],[0.,4.],[3.,3.]],dtype=torch.float64)
for case in data['attention']:
    allowed=torch.tensor(case['allowed'],dtype=torch.bool)
    scores=q@q.T / 2**.5
    weights=scores.masked_fill(~allowed,-torch.inf).softmax(-1)
    compare(case['scores'],scores);compare(case['weights'],weights)
    compare(case['output'],F.scaled_dot_product_attention(q[None,None],q[None,None],v[None,None],attn_mask=allowed,dropout_p=0)[0,0])

rows=torch.tensor([[2.,-2.],[4.,-4.],[0.,0.]],dtype=torch.float64)
cases=iter(data['pooling'])
for mode in ['mean','sum','max']:
    for masked in [True,False]:
        selected=rows[:2] if masked else rows
        expected=selected.mean(0) if mode=='mean' else selected.sum(0) if mode=='sum' else selected.max(0).values
        compare(next(cases)['vector'],expected)
cases=iter(data['quantization'])
for value in [-4,-.625,-.375,0,.375,.625,1.3,4]:
    for scale in [.25,.5]:
        t=torch.tensor(value,dtype=torch.float64);raw=(t/scale).round();integer=raw.clamp(-8,7);case=next(cases)
        compare(case['raw'],raw);compare(case['integer'],integer);compare(case['restored'],integer*scale);compare(case['error'],integer*scale-t)

# Geometry from endpoints, independent of the JavaScript corner implementation.
compare(data['geometry'][0],torch.tensor([20.,15.,30.,30.]))
compare(data['geometry'][1],torch.tensor([100.-20.-30.,15.,30.,30.]))
compare(data['geometry'][2],torch.tensor([0.,(15.-10.)*80/60,30.*100/60,30.*80/60],dtype=torch.float64))

# Check the regenerated raster uses the same genuine valid convolution.
spec=importlib.util.spec_from_file_location('concept_generator',ROOT/'scripts/generate_concept_visuals.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
captured=[]
from matplotlib.axes import Axes
original=Axes.imshow
def capture(axis,array,*args,**kwargs):
    captured.append(torch.as_tensor(array.copy(),dtype=torch.float64))
    return original(axis,array,*args,**kwargs)
with patch.object(Axes,'imshow',capture),patch.object(generator,'save',lambda fig,name:generator.plt.close(fig)):
    generator.convolution()
torch.testing.assert_close(captured[-1],F.conv2d(image,torch.tensor(data['kernels']['edge'],dtype=torch.float64)[None,None])[0,0])
print('Independent CPU numerics passed: SGD, 16 convolution cases, four attention masks, six pooling cases, 16 quantization cases, paired geometry and raster response.')
