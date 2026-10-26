"use strict";
const $ = (id) => document.getElementById(id);
let teams = [], batch = null, fixtureOffset = 0, selectionVersion = 0;
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  return body;
}
function text(id, value) { $(id).textContent = value; }
function option(team) { const el = document.createElement("option"); el.value = team.id; el.textContent = team.name; return el; }
async function refresh() {
  const health = await api("/health");
  text("status", health.model_loaded ? "● Модель готова" : "○ Требуется обучение");
  if (!health.model_loaded) return;
  const metrics = await api("/metrics");
  $("model-rows").replaceChildren();
  for (const [name, scores] of Object.entries(metrics.validation)) {
    const tr = document.createElement("tr");
    for (const value of [name === "logistic_regression" ? "Логистическая регрессия" : "Случайный лес", scores.roc_auc_ovr_macro.toFixed(3), name === metrics.selected_model ? "Выбрана" : "Сравнение"]) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
    $("model-rows").append(tr);
  }
  const comp=metrics.per_competition_test[$("competition").value];
  text("competition-metric",comp ? `ROC-AUC выбранного турнира на тесте: ${comp.roc_auc_ovr_macro.toFixed(3)}. В еврокубках оценены только групповые этапы.` : "Для турнира нет отдельной тестовой оценки.");
  text("test-metric", `Финальный тест ${metrics.test_from} — ${metrics.test_to}: ROC-AUC ${metrics.test.roc_auc_ovr_macro.toFixed(3)}. Обучена по ${metrics.trained_through}. Признаков после отбора: ${metrics.selected_features.length}.`);
}
$("match-form").addEventListener("submit", async (event) => {
  event.preventDefault(); text("error", ""); $("submit").disabled = true;
  try {
    const home = Number($("home").value), away = Number($("away").value);
    const result = await api("/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({matches:[{home_club_id:home,away_club_id:away,date:$("date").value,competition_id:$("competition").value,season:Number($("season").value)}]})});
    const row = result.predictions[0];
    text("match-title", `${teams.find(t=>t.id===home).name} — ${teams.find(t=>t.id===away).name}`);
    for (const name of ["home","draw","away"]) text(`p-${name}`, `${(100*row.probabilities[name]).toFixed(1)}%`);
    text("result-note", row.in_training_period ? "Внимание: дата входит в период обучения. Это не независимый прогноз." : `${result.warning}${["CL","EL","UCOL"].includes($("competition").value) ? " Еврокубки: не вероятность прохода; нейтральное поле не моделируется." : ""}`);
    $("feature-values").replaceChildren();
    for (const [label,key] of [["Разница Elo","elo_diff"],["Очки хозяев / последние 5","home_points5"],["Очки гостей / последние 5","away_points5"],["Отдых хозяев, дней","home_rest"],["Отдых гостей, дней","away_rest"]]) { const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=label;dd.textContent=row.features[key].toFixed(1);$("feature-values").append(dt,dd); }
    $("feature-box").hidden = false;
  } catch (error) { text("error",error.message); } finally { $("submit").disabled=false; }
});
$("swap").addEventListener("click",()=>{const old=$("home").value;$("home").value=$("away").value;$("away").value=old;});
function auth() { return {"X-Admin-Token":$("admin-token").value}; }
$("train").addEventListener("click",async()=>{ $("train").disabled=true;text("admin-message","Обучение…");try{await api("/train",{method:"POST",headers:auth()});text("admin-message","Модель обучена и сохранена");await refresh();}catch(e){text("admin-message",e.message);}finally{$("train").disabled=false;} });
function download(data,name) {const url=URL.createObjectURL(data),a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$("download").addEventListener("click",async()=>{try{const r=await fetch("/model",{headers:auth()});if(!r.ok)throw new Error((await r.json()).detail);download(await r.blob(),"champion.pkl");}catch(e){text("admin-message",e.message);}});
$("upload").addEventListener("click",async()=>{try{if(!$("model-file").files[0])throw new Error("Выберите .pkl");const f=new FormData();f.append("file",$("model-file").files[0]);await api("/upload-model",{method:"POST",headers:auth(),body:f});text("admin-message","Модель загружена");await refresh();}catch(e){text("admin-message",e.message);}});
$("csv-form").addEventListener("submit",async(e)=>{e.preventDefault();text("csv-message","Обработка…");$("csv-download").hidden=true;try{const f=new FormData();f.append("file",$("csv").files[0]);batch=await api("/predict-from-csv",{method:"POST",body:f});text("csv-message",`Матчей: ${batch.rows}. ROC-AUC: ${batch.roc_auc===null ? "не вычислен" : batch.roc_auc.toFixed(3)}. ${batch.metric_note}`);$("csv-download").hidden=false;}catch(err){text("csv-message",err.message);}});
$("csv-download").addEventListener("click",()=>download(new Blob([JSON.stringify(batch,null,2)],{type:"application/json"}),"predictions.json"));
function query() { return `competition_id=${encodeURIComponent($("competition").value)}&season=${$("season").value}`; }
function cells(values) { const tr=document.createElement("tr");for(const value of values){const td=document.createElement("td");td.textContent=value;tr.append(td);}return tr; }
async function loadValues(){
  if(!$("value-club").value)return;
  const club=$("value-club").value, season=$("season").value;
  const data=await api(`/players/valuable?club_id=${club}&season=${season}`);
  if(club!==$("value-club").value || season!==$("season").value)return;
  text("value-date",`Срез на ${data.as_of}; до 10 игроков с доступной оценкой.`);text("value-note",data.note);$("value-rows").replaceChildren();
  for(const p of data.players)$("value-rows").append(cells([`${p.name} · ${p.position}`,new Intl.NumberFormat("ru-RU",{style:"currency",currency:"EUR",maximumFractionDigits:0}).format(p.value_eur),p.valuation_date]));
  if(!data.players.length)$("value-rows").append(cells(["Нет надёжных исторических оценок для клуба на этот срез","—","—"]));
}
async function loadFixtures(){
  const q=query(),offset=fixtureOffset,club=$("only-club").checked ? `&club_id=${$("home").value}` : "";
  const data=await api(`/matches?${q}${club}&limit=20&offset=${offset}`);
  if(q!==query() || offset!==fixtureOffset)return;
  $("fixtures").replaceChildren();
  for(const m of data.matches)$("fixtures").append(cells([`${m.date} / ${m.round}`,`${m.home_club_name} — ${m.away_club_name}`,`${m.home_club_goals}:${m.away_club_goals}`]));
  text("score-note",data.score_note);text("fixture-count",`${data.total ? offset+1 : 0}–${Math.min(offset+20,data.total)} из ${data.total}`);$("previous").disabled=offset===0;$("next").disabled=offset+20>=data.total;
}
async function changeSelection(){
  const version=++selectionVersion;text("error","");fixtureOffset=0;
  try{
    const q=query();const [info,summary]=await Promise.all([api(`/teams?${q}`),api(`/season-summary?${q}`)]);if(version!==selectionVersion)return;
    teams=info.teams;for(const id of ["home","away","value-club"]){$(id).replaceChildren();for(const t of teams)$(id).append(option(t));}
    $("home").value=teams.some(t=>t.id===131)?131:teams[0].id;$("away").value=teams.some(t=>t.id===418)?418:teams[Math.min(1,teams.length-1)].id;$("value-club").value=$("home").value;
    $("date").value=info.scenario_date;const year=Number($("season").value);$("date").min=`${year}-07-01`;$("date").max=`${year+1}-06-30`;
    text("snapshot",`Последний матч турнира в сезоне: ${info.snapshot_through}. Признаки берутся только из предшествующих игр. Для клубов вне пяти лиг внутренняя история неполна.`);
    text("focus-title",$("competition").selectedOptions[0].textContent);text("champion",summary.champion || "Нет подтверждённого результата");text("season-count",`${summary.label} · ${summary.matches} матчей`);text("standings-note",`${summary.method}. ${summary.standings_note}`);
    $("award").replaceChildren();for(const a of summary.awards){const h=document.createElement("h3"),p=document.createElement("p"),link=document.createElement("a");h.textContent=a.name;p.textContent=`${a.club} · ${a.award}`;link.href=a.source;link.textContent="Официальный источник ↗";link.target="_blank";link.rel="noopener noreferrer";$("award").append(h,p,link);}
    if(!summary.awards.length){const p=document.createElement("p");p.textContent="Награда этого сезона пока не внесена в проверенный справочник.";$("award").append(p);}
    $("standings").replaceChildren();for(const t of summary.standings)$("standings").append(cells([t.name,t.played,t.points,`${t.gf}:${t.ga}`]));
    text("match-title","Три исхода. Один матч.");for(const c of ["home","draw","away"])text(`p-${c}`,"—");$("feature-box").hidden=true;
    text("result-note","Выбери команды и дату. Для еврокубков модель обучена на групповых этапах: не прогноз выхода в следующий раунд. Нейтральное поле финалов отдельно не моделируется.");
    await Promise.all([loadFixtures(),loadValues(),refresh()]);
  }catch(e){text("error",e.message);}
}
$("competition").addEventListener("change",changeSelection);$("season").addEventListener("change",changeSelection);
$("value-club").addEventListener("change",()=>loadValues().catch(e=>text("error",e.message)));
$("only-club").addEventListener("change",()=>{fixtureOffset=0;loadFixtures().catch(e=>text("error",e.message));});
$("home").addEventListener("change",()=>{if($("only-club").checked){fixtureOffset=0;loadFixtures().catch(e=>text("error",e.message));}});
$("previous").addEventListener("click",()=>{fixtureOffset=Math.max(0,fixtureOffset-20);loadFixtures().catch(e=>text("error",e.message));});
$("next").addEventListener("click",()=>{fixtureOffset+=20;loadFixtures().catch(e=>text("error",e.message));});
async function init(){try{const data=await api("/competitions");for(const c of data.competitions)$("competition").append(option({id:c.id,name:`${c.name} · ${c.region}`}));for(const s of data.seasons)$("season").append(option({id:s,name:`${s}/${String(s+1).slice(-2)}`}));await changeSelection();}catch(e){text("error",e.message);text("status","Нет подключения к API");}}
init();
