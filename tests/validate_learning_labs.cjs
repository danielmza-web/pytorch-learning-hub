'use strict';
const assert = require('node:assert/strict');
const lab = require('../docs/assets/javascripts/learning-labs.js');
const close = (a, b) => assert.ok(Math.abs(a-b)<1e-10, `${a} != ${b}`);

// Decisions at equality and at the no-alarm boundary have different meanings.
assert.deepEqual(lab.threshold(.4).predicted, [1,1,1,0]);
assert.equal(lab.threshold(1).precision, null);
assert.equal(lab.threshold(0).recall, 1);
for (const value of [NaN, Infinity, -.1, 1.1]) {
  assert.equal(lab.threshold(value), null);
  assert.equal(lab.training(value), null);
  assert.equal(lab.curveScenario(value), null);
}
const zeroRate = lab.training(0);
assert.deepEqual(zeroRate.updated, zeroRate.weights);
close(zeroRate.gradients.flat().reduce((a,b)=>a+b,0), 0);

assert.deepEqual(lab.accumulation(16,3,133), {
  effective:48, updates:3, last:37,
  first:{count:3,full:3,remainder:0}, final:{count:3,full:2,remainder:5}
});
assert.equal(lab.accumulation(16,3,96).last,48);
assert.equal(lab.accumulation(16,3,5).last,5);
for (const values of [[0,3,4],[16,2.5,133],[16,3,NaN],[Number.MAX_SAFE_INTEGER,2,133]])
  assert.equal(lab.accumulation(...values),null);

for (const mode of ['mean','sum','max']) {
  assert.deepEqual(lab.pooling([2,3,0],mode),lab.pooling([3,2,0],mode));
  assert.deepEqual(lab.pooling([0,0],mode),{vector:[0,0],count:0,empty:true});
}
assert.deepEqual(lab.pooling([2,3,0],'max',false).vector,[4,0]);
assert.deepEqual(lab.pooling([2,3,0],'max',true).vector,[4,-2]);
assert.equal(lab.pooling([99],'mean'),null);
assert.equal(lab.pooling([],'mean'),null);
assert.equal(lab.connection('mobile',16,32).weights,656);
assert.equal(lab.connection('mobile',16,32).denseWeights,4608);
assert.equal(lab.connection('residual',16,32).channels,16);
assert.equal(lab.connection('dense',16,32).channels,48);
assert.equal(lab.connection('mobile',1.5,32),null);

for (const kind of ['encoder','causal']) for (const padding of [false,true]) {
  const a=lab.attention(kind,padding);
  a.weights.forEach((row,r)=>{
    close(row.reduce((s,v)=>s+v,0),1);
    row.forEach((v,c)=>{if(!a.allowed[r][c])assert.equal(v,0);});
  });
}
assert.deepEqual(lab.attention('causal').output[0],[2,0]);
assert.equal(lab.attention('bad'),null);
assert.equal(lab.quantize(4).clipped,true);
assert.equal(lab.quantize(-4).integer,-8);
assert.equal(lab.quantize(.625).integer,2); // tie to even
assert.equal(lab.quantize(-.625).integer,-2);
assert.equal(lab.quantize(1,0),null);
assert.equal(lab.quantize(NaN),null);
assert.deepEqual(lab.transformBox([20,15,30,30],'flip'),[50,15,30,30]);
assert.equal(lab.transformBox([0,0,10,5],'crop'),null);
const weak=lab.curveScenario(0), strong=lab.curveScenario(1), balanced=lab.curveScenario(.5);
assert.ok(weak.validation.at(-1)>weak.validation[20]);
assert.ok(weak.train.at(-1)<weak.train[0]);
assert.ok(strong.train.at(-1)>balanced.train.at(-1)+.5);
assert.ok(strong.validation.at(-1)>balanced.validation.at(-1)+.5);
assert.ok(balanced.validation.every((v,i,a)=>!i||v<a[i-1]));
for (const strength of [.25,.35,.5,.78]) {
  const a=lab.curveScenario(strength);
  assert.ok(a.validation.every((v,i,values)=>!i||v<values[i-1]));
}
console.log('Learning lab boundaries passed: equality, empty padding, invalid numbers, masks, rounding and coherent loss scenarios.');
