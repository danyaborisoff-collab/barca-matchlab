"use strict";
const $ = (id) => document.getElementById(id);
let teams = [], batch = null;
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
  text("test-metric", `Финальный тест ${metrics.test_from} — ${metrics.test_to}: ROC-AUC ${metrics.test.roc_auc_ovr_macro.toFixed(3)}. Обучена по ${metrics.trained_through}. Признаков после отбора: ${metrics.selected_features.length}.`);
}
$("match-form").addEventListener("submit", async (event) => {
  event.preventDefault(); text("error", ""); $("submit").disabled = true;
  try {
    const home = Number($("home").value), away = Number($("away").value);
    const result = await api("/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({matches:[{home_club_id:home,away_club_id:away,date:$("date").value}]})});
    const row = result.predictions[0];
    text("match-title", `${teams.find(t=>t.id===home).name} — ${teams.find(t=>t.id===away).name}`);
    for (const name of ["home","draw","away"]) text(`p-${name}`, `${(100*row.probabilities[name]).toFixed(1)}%`);
    text("result-note", row.in_training_period ? "Внимание: дата входит в период обучения. Это не независимый прогноз." : result.warning);
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
async function init(){try{const info=await api("/teams");teams=info.teams;for(const t of teams){$("home").append(option(t));$("away").append(option(t));}$("home").value=131;$("away").value=418;$("date").value=info.scenario_date;text("snapshot",`Данные заканчиваются ${info.snapshot_through}. Более поздняя дата не добавляет новые матчи.`);await refresh();}catch(e){text("error",e.message);text("status","Нет подключения к API");}}
init();
