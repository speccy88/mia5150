// Offline DOM-contract simulation, not a browser screenshot/visual test.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const payload=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];const data=JSON.parse(payload);
assert(!/<(?:script|img|link)[^>]+(?:src|href)\s*=/i.test(html));
assert(!/\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(/.test(script));
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.value='';this.textContent='';this.attrs={};}
 appendChild(el){this.children.push(el);return el;}append(...els){this.children.push(...els);}replaceChildren(...els){this.children=els;}
 addEventListener(name,fn){this.events[name]=fn;}setAttribute(k,v){this.attrs[k]=v;}click(){this.events.click();}
}
const elements=new Map();const doc={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(tag){return new Element(tag);}};
doc.getElementById('data').textContent=payload;const context=vm.createContext({document:doc,JSON,Number,String});vm.runInContext(script,context);
let states=0;
data.panels.forEach((panel,pi)=>{
 const cohort=doc.getElementById('cohort');cohort.value=String(pi);cohort.events.change();
 panel.rows.forEach((row,qi)=>{
  const query=doc.getElementById('query');query.value=String(qi);query.events.change();
  [5,10].forEach(k=>{
   const cutoff=doc.getElementById('cutoff');cutoff.value=String(k);cutoff.events.change();
   assert.equal(doc.getElementById('task').textContent,row.query);
   assert.equal(doc.getElementById('baseline').children.filter(e=>e.tag==='li').length,Math.min(k,row.baseline.length));
   assert.equal(doc.getElementById('reranked').children.filter(e=>e.tag==='li').length,row.status==='ok'?Math.min(k,row.reranked.length):0);
   assert(doc.getElementById('metrics').textContent.includes('oracle Recall5'));
   doc.getElementById('failure').click();assert(doc.getElementById('metrics').textContent.includes('Strict Jev rerank: Recall5 0.0000'));
   assert(doc.getElementById('metrics').textContent.includes('Served retrieval fallback'));
   assert.equal(doc.getElementById('reranked').children.filter(e=>e.tag==='li').length,0);
   assert.equal(doc.getElementById('fusion').children.filter(e=>e.tag==='li').length,Math.min(k,row.baseline.length));
   doc.getElementById('failure').click();states+=2;
  });
 });
});
console.log(JSON.stringify({panels:data.panels.length,display_cases:64,interaction_states_checked:states,network_calls:0,dom_contract:'passed',visual_browser_render:'not verified'}));
