"""Browser UI of the audit panel (HTML+JS string, mounted via annotate_audit.HTML)."""
HTML = """
<div class="card" id="audit"><b>Аудит</b> <span id="u_prog" class="mut"></span>
<div><button id="u_load">Загрузить моменты</button> <button id="u_prev">Назад</button> <button id="u_next">Далее</button>
 <button id="u_r1btn">Роль в клипах</button></div>
<video id="uv" controls style="margin-top:8px"></video><div id="u_q"></div></div>
<script>
(()=>{const g=id=>document.getElementById(id);
const J=async(u,o)=>{const r=await fetch(u,o);if(!r.ok)throw new Error(r.status);return r.json()};
const P=(u,b)=>J(u,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
let Q,items=[],cur=0,ans={},t1=null,v1=null,mode="m",cid="";
const btn=(on,attrs,label)=>`<button ${attrs} style="margin:2px;${on?'':'background:var(--card);color:var(--fg);border:1px solid var(--mut)'}">${label}</button>`;
function seek(clip,t){const v=g("uv"),src=`/api/c/${cid}/video/${clip}`;const f=()=>{v.currentTime=t};
 if(v.dataset.src!==src){v.dataset.src=src;v.src=src;v.onloadedmetadata=f}else f()}
function prog(){const a=items.filter(x=>mode==="m"?x.answers:x.r1).length;g("u_prog").textContent=`сохранено ${a} из ${items.length}, осталось ${items.length-a}`}
function show(){if(!items.length)return;const it=items[cur];prog();seek(it.clip,it.t);
 if(mode==="m"){const d=it.draft||{a:it.answers||{},t:it.t1||null,v:it.v1||null};ans=d.a;t1=d.t;v1=d.v;
  g("u_q").innerHTML=`<div><b>${Q.t1.text}</b><br>`+Q.t1.options.map(o=>btn(t1===o.key,`data-t="${o.key}"`,o.label)).join("")+"</div>"+`<div style="margin:6px 0"><b>${Q.v1.text}</b><br>`+Q.v1.options.map(o=>btn(v1===o.key,`data-v="${o.key}"`,o.label)).join("")+"</div>"+
   Q.q.map(q=>`<div style="margin:6px 0"><b>${q.text}</b><br>`+q.options.map(o=>btn(ans[q.id]===o.key,`data-q="${q.id}" data-k="${o.key}"`,o.label)).join("")+"</div>").join("")}
 else g("u_q").innerHTML=`<div><b>${Q.r1.text}</b><br>`+Q.r1.options.map(o=>btn(it.r1===o.key,`data-r="${o.key}"`,o.label)).join("")+"</div>"}
async function advance(){const n=items.findIndex((x,i)=>i>cur&&!(mode==="m"?x.answers:x.r1));if(n>=0)cur=n;else if(cur<items.length-1)cur++;show()}
g("u_q").onclick=async e=>{const t=e.target,it=items[cur];
 if(mode==="m"){if(t.dataset.t)t1=t.dataset.t;else if(t.dataset.v)v1=t.dataset.v;else if(t.dataset.q)ans[t.dataset.q]=t.dataset.k;else return;
  it.draft={a:ans,t:t1,v:v1};
  if(t1&&v1&&Object.keys(ans).length===Q.q.length){await P(`/api/c/${cid}/audit/answer/${it.pid}`,{answers:ans,t1,v1});it.answers=Object.assign({},ans);it.t1=t1;it.v1=v1;it.draft=null;await advance()}else show()}
 else if(t.dataset.r){await P(`/api/c/${cid}/audit/r1/${it.pid}`,{r1:t.dataset.r});it.r1=t.dataset.r;await advance()}};
async function load(m){cid=g("cid").value;mode=m;try{Q=Q||await J("/api/audit/questions");
 items=(await J(m==="m"?`/api/c/${cid}/audit/moments`:`/api/c/${cid}/audit/r1`)).items;
 const f=items.findIndex(x=>!(m==="m"?x.answers:x.r1));cur=f<0?0:f;show()}catch(e){g("u_prog").textContent="недоступно"}}
g("u_load").onclick=()=>load("m");g("u_r1btn").onclick=()=>load("r");
g("u_next").onclick=()=>{if(cur<items.length-1){cur++;show()}};g("u_prev").onclick=()=>{if(cur>0){cur--;show()}};
})();
</script>
"""
