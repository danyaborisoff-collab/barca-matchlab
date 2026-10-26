import asyncio
import io
import logging
import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.artifacts import MAX_BYTES, ROOT, loads, model_dir, save
from ml.features import FEATURES, LABELS, load_games, raw_features
from ml.train import train

log = logging.getLogger("matchlab")


@asynccontextmanager
async def lifespan(app):
    app.state.games = load_games()
    app.state.bundle = None
    app.state.train_lock = asyncio.Lock()
    path = model_dir() / "champion.pkl"
    if path.exists():
        try:
            app.state.bundle = loads(path.read_bytes())
        except (ValueError, EOFError):
            log.exception("Сохранённая модель отклонена")
    yield


app = FastAPI(title="MatchLab · La Liga", version="0.7.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.middleware("http")
async def timing(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.6f}"
    log.info(
        "%s %s status=%s seconds=%.6f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


def admin(x_admin_token: str = Header(default="")):
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Операции управления отключены: задайте ADMIN_TOKEN")
    if not secrets.compare_digest(expected, x_admin_token):
        raise HTTPException(401, "Неверный X-Admin-Token")


def bundle():
    value = app.state.bundle
    if value is None:
        raise HTTPException(
            503, "Модель не загружена. Выполните обучение или загрузите подписанную модель"
        )
    return value


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home_club_id: int = Field(gt=0)
    away_club_id: int = Field(gt=0)
    date: date

    @model_validator(mode="after")
    def different(self):
        if self.home_club_id == self.away_club_id:
            raise ValueError("Команды должны различаться")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: list[Match] | None = Field(default=None, min_length=1, max_length=100)
    features: list[dict[str, float]] | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def one_mode(self):
        if (self.matches is None) == (self.features is None):
            raise ValueError("Передайте только matches или только features")
        if self.features:
            for row in self.features:
                if set(row) != set(FEATURES) or not all(math.isfinite(v) for v in row.values()):
                    raise ValueError("Нужны все конечные числовые признаки из GET /features")
        return self


def predictions(features, original, dates=None):
    artifact = bundle()
    pipe = artifact["pipeline"]
    prob = pipe.predict_proba(pd.DataFrame(features)[FEATURES])
    classes = list(pipe.classes_)
    result = []
    for i, row in enumerate(original):
        probabilities = {c: float(prob[i, classes.index(c)]) for c in LABELS}
        item = {
            **row,
            "features": features[i],
            "predicted_result": max(probabilities, key=probabilities.get),
            "probabilities": probabilities,
        }
        if dates:
            item["in_training_period"] = dates[i] <= artifact["report"]["trained_through"]
        result.append(item)
    return result


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.bundle is not None}


@app.get("/teams")
def teams():
    df = app.state.games
    names = {}
    for row in df.itertuples():
        names[int(row.home_club_id)] = row.home_club_name
        names[int(row.away_club_id)] = row.away_club_name
    active = set(df[df.season == 2024].home_club_id)
    return {
        "teams": [
            {"id": i, "name": n, "latest_season": i in active}
            for i, n in sorted(names.items(), key=lambda t: t[1])
        ],
        "snapshot_through": str(df.date.max().date()),
        "scenario_date": str(df.date.max().date() + timedelta(days=1)),
        "barcelona_id": 131,
    }


@app.get("/features")
def features():
    return {"required_features": FEATURES, "result_classes": LABELS}


@app.get("/metrics")
def metrics():
    return bundle()["report"]


@app.post("/predict")
def predict(request: PredictRequest):
    bundle()
    if request.features is not None:
        return {
            "predictions": predictions(request.features, [{} for _ in request.features]),
            "warning": "Признаки клиента должны быть рассчитаны только по прошлым матчам",
        }
    try:
        original = [r.model_dump(mode="json") for r in request.matches]
        feats = [
            raw_features(app.state.games, r.home_club_id, r.away_club_id, r.date)
            for r in request.matches
        ]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "predictions": predictions(feats, original, [r["date"] for r in original]),
        "warning": "Историческая модель; не учитывает актуальные составы, травмы и трансферы. Не рекомендация для ставок",
    }


@app.post("/predict-from-csv")
async def predict_csv(file: UploadFile = File(...)):
    bundle()
    data = await file.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise HTTPException(413, "CSV больше 2 MiB")
    try:
        frame = pd.read_csv(io.BytesIO(data))
        if not 1 <= len(frame) <= 500:
            raise ValueError("Допустимо от 1 до 500 строк")
        if set(FEATURES).issubset(frame.columns):
            feats = frame[FEATURES].apply(pd.to_numeric, errors="raise").to_dict("records")
            PredictRequest(features=feats[:100])
            if not all(math.isfinite(v) for r in feats for v in r.values()):
                raise ValueError("Признаки должны быть конечными числами")
            outputs = predictions(feats, frame.fillna("").to_dict("records"))
        else:
            rows = [
                Match.model_validate(r)
                for r in frame[["home_club_id", "away_club_id", "date"]].to_dict("records")
            ]
            feats = [
                raw_features(app.state.games, r.home_club_id, r.away_club_id, r.date) for r in rows
            ]
            outputs = predictions(
                feats, frame.fillna("").to_dict("records"), [r.date.isoformat() for r in rows]
            )
        auc, note = None, "Без result ROC-AUC не вычисляется"
        if "result" in frame:
            if not set(frame.result).issubset(LABELS):
                raise ValueError("result должен быть home, draw или away")
            if set(frame.result) == set(LABELS):
                from sklearn.metrics import roc_auc_score

                auc = float(
                    roc_auc_score(
                        frame.result,
                        [[o["probabilities"][c] for c in LABELS] for o in outputs],
                        labels=LABELS,
                        multi_class="ovr",
                        average="macro",
                    )
                )
                note = "ROC-AUC macro OVR по вероятностям. Это оценка загруженного CSV, не обязательно независимый тест"
            else:
                note = "Для macro ROC-AUC нужны все три класса; возвращён null"
    except (ValueError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"rows": len(outputs), "roc_auc": auc, "metric_note": note, "dataset": outputs}


@app.post("/train", dependencies=[Depends(admin)])
async def train_model():
    if app.state.train_lock.locked():
        raise HTTPException(409, "Обучение уже выполняется")
    async with app.state.train_lock:
        app.state.bundle = await asyncio.to_thread(train)
    return {"status": "trained", "metrics": app.state.bundle["report"]}


@app.post("/upload-model", dependencies=[Depends(admin)])
async def upload_model(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pkl"):
        raise HTTPException(422, "Ожидается .pkl")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Модель больше 25 MiB")
    try:
        value = loads(data)
        if value["report"]["features"] != FEATURES:
            raise ValueError("Несовместимая схема признаков")
    except (ValueError, KeyError, EOFError) as exc:
        raise HTTPException(422, str(exc)) from exc
    save(value)
    app.state.bundle = value
    return {"status": "loaded", "model": value["report"]["selected_model"]}


@app.get("/model", dependencies=[Depends(admin)])
def download_model():
    bundle()
    return FileResponse(
        model_dir() / "champion.pkl", filename="champion.pkl", media_type="application/octet-stream"
    )


@app.get("/example-csv")
def example_csv():
    path = ROOT / "data/example_raw.csv"
    if not path.exists():
        raise HTTPException(404, "Пример появится после обучения")
    return FileResponse(path, filename=path.name, media_type="text/csv")
