"""HTML/JS for tools/annotate_intervals.py (single page, no external resources, Russian, enums + times only)."""
PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Интервалы</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#f4f4f6;color:#1c1c20}
header{padding:8px 14px;background:#fff;border-bottom:1px solid #ccc;position:sticky;top:0;z-index:2}
main{display:flex;gap:14px;padding:12px;flex-wrap:wrap}
#list{width:230px}#work{flex:1;min-width:420px}
.t{display:block;width:100%;text-align:left;margin:2px 0;padding:6px;border:1px solid #bbb;background:#fff;cursor:pointer;border-radius:4px}
.t.cur{border-color:#06c;background:#e8f1ff}.t.ctl{margin-left:18px;width:calc(100% - 18px)}
video{width:100%;max-height:50vh;background:#000}
.iv{background:#fff;border:1px solid #ccc;border-radius:4px;padding:6px;margin:6px 0}
.iv label{margin-right:10px;white-space:nowrap}input[type=number]{width:80px}
#st{font-weight:600}#st.err{color:#b00}.box{background:#fff;border:1px solid #ccc;padding:8px;margin:8px 0;border-radius:4px}
small{color:#555}
</style></head><body>
<header><b>Разметка интервалов (локально)</b> &middot; <span id="st">...</span><br><small id="cnt"></small></header>
<main><div id="list"></div><div id="work"><p>Выберите ролик слева.</p></div></main>
<script>
const CID="__CID__",B="/api/i/"+CID;
const $=id=>document.getElementById(id);
const VIS={good:"хорошо видно",partial:"частично",poor:"плохо видно"},CF={sure:"уверен",unsure:"не уверен"};
const MSG={unknown_with_intervals:"Удалите интервалы или снимите unknown",need_interval_or_flag:"Нужен интервал или отметка",
 unfinished_interval:"Завершите начатый интервал",bad_times:"Проверьте время начала/конца",need_confirmations:"Нужны обе отметки подтверждения",
 answer_and_intervals:"Уберите интервалы или ответ",previous_control_unfinished:"Сначала завершите предыдущий контроль",
 not_needed:"Контроль уже подтверждён",no_candidate:"Кандидатов больше нет",limit_per_positive:"Лимит контролей исчерпан"};
let S=null,tm=null;   // S = state of the OPEN task: {id,role,rev,pl,dur,dirty,blocked,chain}
function st(t,e){const s=$("st");s.textContent=t;s.className=e?"err":""}
async function api(m,u,b){const o={method:m,headers:{"Content-Type":"application/json"}};if(b!==undefined)o.body=JSON.stringify(b);
 let r;try{r=await fetch(u,o)}catch(e){return {s:0,j:{}}}
 let j={};try{j=await r.json()}catch(e){}return {s:r.status,j}}
const dk=id=>"ivd:"+CID+":"+id;
function lsGet(k){try{return JSON.parse(localStorage.getItem(k))}catch(e){return null}}
function lsSet(k,v){try{v===null?localStorage.removeItem(k):localStorage.setItem(k,JSON.stringify(v))}catch(e){}}
function cnt(c){const p=c.positives,k=c.controls;
 $("cnt").textContent=`Нарушения: показано ${p.shown}, завершено с интервалами ${p.completed_with_intervals}, завершено «неизвестно» ${p.completed_unknown}, всего ${p.total}. `+
 `Контроль: сгенерировано ${k.generated} из максимум ${k.max}; показано ${k.shown}; подтверждено нормальных уверенно ${k.confirmed_normal_confident}, с низкой уверенностью ${k.confirmed_normal_low_confidence}; не сопоставимо/неизвестно ${k.not_comparable_or_unknown}.`}
async function loadList(){const r=await api("GET",B+"/tasks");if(r.s!=200){st("Ошибка: список",1);return}
 cnt(r.j.counters);const L=$("list");L.innerHTML="";
 r.j.tasks.forEach(t=>{const b=document.createElement("button");b.className="t"+(t.role=="control"?" ctl":"")+(S&&S.id==t.task_id?" cur":"");
  const nm=t.role=="positive"?"Ролик "+t.ord:"Контроль к ролику "+t.ord+", кандидат "+(t.cand||"?")+"/"+(t.of||"?");
  b.textContent=nm+(t.shown?" · показан":"")+(t.state?" · завершён":"");b.onclick=()=>openTask(t.task_id);L.appendChild(b)})}
// ---- serialized save queue (per open task; snapshots; late replies only touch their own task state)
function enqueue(T,fn){T.chain=T.chain.then(fn,fn);return T.chain}
function fail(T,r){T.blocked=true;
 if(r.s==409)conflict(T,{rev:T.rev,payload:T.pl},r.j.current_rev);
 else if(r.s==0){st("Не сохранено — черновик сохранён локально",1);setTimeout(()=>{if(S===T&&!T.fatal){T.blocked=false;run(T)}},3000)}
 else if(r.s==422)st("Ошибка: "+(MSG[r.j.detail]||"не сохранено")+" (черновик сохранён локально)",1);
 else st("Ошибка: "+r.s+" (черновик сохранён локально)",1)}
// ---- conflict: never auto-discard; backup in a separate key; two explicit choices
async function conflict(T,draft,srvRev){T.fatal=true;T.blocked=true;
 lsSet("ivb:"+CID+":"+T.id,{rev:draft.rev,payload:draft.payload,ts:Date.now()});
 const g=await api("GET",B+"/task/"+T.id);const N=g.s==200?g.j.rev:srvRev;
 st("Конфликт: сервер rev "+N+", локальный черновик основан на rev "+draft.rev,1);
 if(S!==T)return;
 const c=$("cf");c.hidden=false;c.innerHTML=`<b>Конфликт версий.</b> Сервер: rev ${N}. Ваш черновик: rev ${draft.rev} (резервная копия сохранена локально).<br>
 <button id="cf_s">Загрузить серверную версию</button> <button id="cf_m">Восстановить мой черновик поверх rev ${N}</button>
 <small id="cf_w" hidden> Внимание: это заменит данные на сервере вашим черновиком и увеличит rev. Нажмите ещё раз для подтверждения.</small>`;
 render();
 $("cf_s").onclick=async()=>{const r=await api("GET",B+"/task/"+T.id);if(r.s!=200)return;T.rev=r.j.rev;T.pl=r.j.payload||T.pl;T.dirty=false;T.blocked=false;T.fatal=false;
  lsSet(dk(T.id),null);c.hidden=true;render();st("Загружена серверная версия rev "+T.rev)};
 let armed=false;$("cf_m").onclick=async()=>{if(!armed){armed=true;$("cf_w").hidden=false;return}
  const r=await api("GET",B+"/task/"+T.id);if(r.s!=200)return;T.rev=r.j.rev;T.pl=draft.payload;T.pl.viewed=!!(r.j.payload&&r.j.payload.viewed);T.dirty=true;T.blocked=false;T.fatal=false;
  c.hidden=true;render();run(T)}}
function run(T){return enqueue(T,async()=>{if(!T.dirty||T.fatal||T.blocked)return;
 const snap=JSON.stringify(T.pl);const r=await api("PUT",B+"/task/"+T.id,{rev:T.rev,payload:JSON.parse(snap)});
 if(r.s==200){T.rev=r.j.rev;if(JSON.stringify(T.pl)==snap){T.dirty=false;lsSet(dk(T.id),null)}else lsSet(dk(T.id),{rev:T.rev,payload:T.pl});
  if(S===T){st("Сохранено rev "+T.rev);cnt(r.j.counters);loadList()}if(T.dirty)run(T)}
 else if(S===T)fail(T,r);else{T.blocked=true}})}
function save(){const T=S;if(!T||T.fatal)return;T.dirty=true;T.blocked=false;lsSet(dk(T.id),{rev:T.rev,payload:T.pl});
 st("Сохранение...");clearTimeout(tm);tm=setTimeout(()=>run(T),350)}
async function flushNow(T){clearTimeout(tm);await run(T);await T.chain}
async function openTask(id){
 if(S){if(S.id==id)return;await flushNow(S);if(S.dirty||S.blocked){st("Ошибка: сначала исправьте сохранение текущего ролика",1);return}}
 const r=await api("GET",B+"/task/"+id);if(r.s!=200){st("Ошибка: задача",1);return}
 const role=r.j.role;let pl=r.j.payload||(role=="positive"?{intervals:[],clip_flags:{not_visible:false,unknown:false,event_unconfirmed:false},same_clip_controls:[],control_not_found:false,done:false}:{intervals:[],answer:null,done:false});
 const T={id,role,ord:r.j.ord,parent:r.j.parent,cand:r.j.cand,of:r.j.of,rev:r.j.rev,pl,dur:r.j.duration||0,dirty:false,blocked:false,fatal:false,chain:Promise.resolve()};
 const d=lsGet(dk(id));let note="";
 if(d&&d.payload&&JSON.stringify(d.payload)!=JSON.stringify(pl)){
  if(d.rev==T.rev){T.pl=d.payload;T.dirty=true;T.restored=true;note=" (восстановлен локальный черновик)"}
  else T.stale=d}
 S=T;window.__S=T;
 $("work").innerHTML='<div id="hd"></div><div id="vid"></div><div id="cf" class="box" hidden></div><div id="form"></div>';
 const v=document.createElement("video");v.id="v";v.controls=true;v.preload="metadata";$("vid").appendChild(v);   // created ONCE per task
 v.addEventListener("loadeddata",()=>ev(T,"viewed"),{once:true});
 v.addEventListener("error",()=>ev(T,"media_error"),{once:true});
 v.src=B+"/video/"+id;
 $("hd").innerHTML=role=="positive"?`<b>Ролик ${T.ord}</b>`:`<b>Контроль к ролику ${T.ord}, кандидат ${T.cand}/${T.of}</b> <button id="back">Вернуться к ролику ${T.ord}</button>`;
 if($("back"))$("back").onclick=()=>openTask(T.parent);
 st((T.rev?"Сохранено rev "+T.rev:"Новый")+note,T.fatal?1:0);render();loadList();if(T.stale)conflict(T,T.stale,T.rev);else if(T.dirty)run(T)}
function ev(T,kind){const key=kind=="viewed"?"viewed":"media_error";
 if(T.pl[key])return;
 enqueue(T,async()=>{if(T.pl[key]||T.fatal)return;const r=await api("POST",B+"/task/"+T.id+"/"+kind,{rev:T.rev});
  if(r.s==200){T.rev=r.j.rev;T.pl[key]=true;if(S===T)loadList()}
  else if(r.s==409){if(S===T)conflict(T,{rev:T.rev,payload:T.pl},r.j.current_rev)}
  else if(S===T)st("Ошибка: "+r.s,1)})}
function ivHtml(list,key,pos){return list.map((v,i)=>`<div class="iv" data-k="${key}" data-i="${i}">
 Начало <input type="number" step="0.1" min="0" data-f="start" value="${v.start??""}" ${v.start_unknown?"disabled":""}>
 <button data-a="cs">Начало = текущее</button>
 Конец <input type="number" step="0.1" min="0" data-f="end" value="${v.end??""}" ${v.end_unknown?"disabled":""}>
 <button data-a="ce">Конец = текущее</button> <button data-a="del">Удалить</button><br>
 ${pos?`<label><input type="checkbox" data-f="start_uncertain" ${v.start_uncertain?"checked":""}>начало неточно</label>
 <label><input type="checkbox" data-f="end_uncertain" ${v.end_uncertain?"checked":""}>конец неточно</label>
 <label><input type="checkbox" data-f="start_unknown" ${v.start_unknown?"checked":""}>начало неизвестно</label>
 <label><input type="checkbox" data-f="end_unknown" ${v.end_unknown?"checked":""}>конец неизвестно</label>`:
 `<label><input type="checkbox" data-f="comparable_visible" ${v.comparable_visible?"checked":""}>Сопоставимая операция присутствует и видна</label>
 <label><input type="checkbox" data-f="no_violation" ${v.no_violation?"checked":""}>Нарушения в ней не видно</label>`}<br>
 Видимость <select data-f="visibility">${Object.keys(VIS).map(k=>`<option value="${k}" ${v.visibility==k?"selected":""}>${VIS[k]}</option>`).join("")}</select>
 Уверенность <select data-f="confidence">${Object.keys(CF).map(k=>`<option value="${k}" ${v.confidence==k?"selected":""}>${CF[k]}</option>`).join("")}</select></div>`).join("")}
function render(){const pl=S.pl,pos=S.role=="positive";   // re-renders ONLY #form; the <video> element is never touched
 let h=`<p><small>${pos?"Отметьте интервал(ы) нарушения: поставьте паузу на нужном кадре и нажмите кнопку «текущее». Если границу не видно — отметьте «неизвестно», числа выдумывать не нужно.":
 "Контрольный ролик: отметьте сопоставимую операцию без нарушения и подтвердите обе галочки, либо выберите «не сопоставимо / не видно / unknown»."}</small></p>`;
 if(pos){h+=`<div class="box"><b>Нарушение</b>${ivHtml(pl.intervals,"intervals",true)}<button id="add">Добавить интервал</button><br>
 <label><input type="checkbox" data-c="not_visible" ${pl.clip_flags.not_visible?"checked":""}>Не видно</label>
 <label><input type="checkbox" data-c="unknown" ${pl.clip_flags.unknown?"checked":""}>Неизвестно</label>
 <label><input type="checkbox" data-c="event_unconfirmed" ${pl.clip_flags.event_unconfirmed?"checked":""}>Событие не подтверждено</label></div>
 <div class="box"><b>Контроль в этом же ролике</b>${ivHtml(pl.same_clip_controls,"same_clip_controls",false)}<button id="addc">Добавить интервал контроля</button><br>
 <button id="nf">Контроль в этом клипе не найден</button> <small id="nfs">${pl.control_not_found?"Следующий контрольный ролик доступен в списке слева.":""}</small></div>`}
 else{h+=`<div class="box"><b>Контроль</b>${ivHtml(pl.intervals,"intervals",false)}<button id="add">Добавить интервал</button><br>
 Ответ <select id="ans">${[["","интервал(ы) выше"],["not_comparable","не сопоставимо"],["not_visible","не видно"],["unknown","unknown"]].map(a=>`<option value="${a[0]}" ${(pl.answer||"")==a[0]?"selected":""}>${a[1]}</option>`).join("")}</select></div>`}
 h+=`<div class="box"><label><input type="checkbox" id="done" ${pl.done?"checked":""}><b>Ролик завершён</b></label></div>`;
 $("form").innerHTML=`<fieldset ${S.fatal?"disabled":""} style="border:0;margin:0;padding:0">${h}</fieldset>`;wire()}
const cts=()=>Math.round($("v").currentTime*1000)/1000;
function newIv(pos){return pos?{start:null,end:null,start_unknown:false,end_unknown:false,start_uncertain:false,end_uncertain:false,visibility:"good",confidence:"sure"}:{start:null,end:null,visibility:"good",confidence:"sure",comparable_visible:false,no_violation:false}}
function wire(){const W=$("form"),pl=S.pl;
 W.querySelectorAll(".iv").forEach(d=>{const L=pl[d.dataset.k],v=L[+d.dataset.i];
  d.querySelectorAll("[data-f]").forEach(e=>e[e.type=="number"?"oninput":"onchange"]=()=>{const f=e.dataset.f;
   if(e.type=="checkbox"){v[f]=e.checked;if(f=="start_unknown"&&e.checked)v.start=null;if(f=="end_unknown"&&e.checked)v.end=null;if(f.endsWith("unknown"))render()}
   else if(e.tagName=="SELECT")v[f]=e.value;else v[f]=e.value===""?null:parseFloat(e.value);save()});
  d.querySelector('[data-a=cs]').onclick=()=>{v.start=cts();v.start_unknown=false;render();save()};
  d.querySelector('[data-a=ce]').onclick=()=>{v.end=cts();v.end_unknown=false;render();save()};
  d.querySelector('[data-a=del]').onclick=()=>{L.splice(+d.dataset.i,1);render();save()}});
 const ad=$("add");if(ad)ad.onclick=()=>{pl.intervals.push(newIv(S.role=="positive"));render();save()};
 const ac=$("addc");if(ac)ac.onclick=()=>{pl.same_clip_controls.push(newIv(false));render();save()};
 W.querySelectorAll("[data-c]").forEach(e=>e.onchange=()=>{pl.clip_flags[e.dataset.c]=e.checked;save()});
 const a=$("ans");if(a)a.onchange=()=>{pl.answer=a.value||null;save()};
 $("done").onchange=e=>{pl.done=e.target.checked;save()};
 const nf=$("nf");if(nf)nf.onclick=async()=>{const T=S;if(T.fatal)return;pl.control_not_found=true;T.dirty=true;T.blocked=false;lsSet(dk(T.id),{rev:T.rev,payload:T.pl});
  await flushNow(T);if(T.dirty||T.blocked){$("nfs").textContent="Контроль не запрошен: сначала исправьте сохранение";return}
  const r=await api("POST",B+"/task/"+T.id+"/next_control");
  $("nfs").textContent=r.s==200?"Контрольный ролик добавлен в список слева.":"Контрольный ролик недоступен: "+(MSG[r.j.detail]||r.s);loadList()}}
window.addEventListener("beforeunload",()=>{if(S&&S.dirty)lsSet(dk(S.id),{rev:S.rev,payload:S.pl})});
loadList();
</script></body></html>"""
