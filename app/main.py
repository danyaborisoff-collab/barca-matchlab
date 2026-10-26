import asyncio
import io
import json
import logging
import math
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.artifacts import MAX_BYTES, ROOT, loads, model_dir, save
from ml.features import COMPETITIONS, FEATURES, LABELS, load_games, raw_features
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
            value = loads(path.read_bytes())
            if (
                value["report"].get("schema_version") != 2
                or value["report"]["features"] != FEATURES
            ):
                raise ValueError("Модель v0.7 несовместима: выполните обучение v0.8")
            app.state.bundle = value
        except (ValueError, KeyError, EOFError):
            log.exception("Сохранённая модель отклонена")
    yield


app = FastAPI(title="MatchLab · European Football", version="0.8.0", lifespan=lifespan)
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


@lru_cache(maxsize=8)
def catalog(name="catalog"):
    return json.loads((ROOT / f"data/{name}.json").read_text(encoding="utf-8"))


def get_season(competition_id, season):
    for row in catalog()["seasons"]:
        if row["competition_id"] == competition_id and row["season"] == season:
            return row
    raise HTTPException(404, "Турнир или сезон отсутствует в архиве")


def match_features(row):
    # Without an explicit season, use the season preceding the scenario date.
    season = (
        row.season
        if row.season is not None
        else min(2024, row.date.year if row.date.month >= 7 else row.date.year - 1)
    )
    info = get_season(row.competition_id, season)
    if row.home_club_id not in info["club_ids"] or row.away_club_id not in info["club_ids"]:
        raise ValueError("Обе команды должны участвовать в выбранном турнире и сезоне")
    if row.date < date(season, 7, 1) or row.date > date(season + 1, 6, 30):
        raise ValueError("Дата сценария должна относиться к выбранному сезону")
    return raw_features(
        app.state.games, row.home_club_id, row.away_club_id, row.date, row.competition_id
    )


@app.get("/competitions")
def competitions():
    return {"competitions": catalog()["competitions"], "seasons": list(range(2024, 2011, -1))}


@app.get("/season-summary")
def season_summary(competition_id: str = "ES1", season: int = 2024):
    info = get_season(competition_id, season)
    awards = [
        a
        for a in catalog("awards")
        if a["competition_id"] == competition_id and a["season"] == season
    ]
    return {
        **info,
        "awards": awards,
        "award_note": "Только подтверждённые награды. Отсутствие записи не означает отсутствие награды; архив не содержит голов и передач игроков.",
        "standings_note": "Расчётная таблица, не официальный протокол. Ligue 1 2019/20 завершена досрочно. Нет учёта санкций и специальных тай-брейков.",
    }


@app.get("/matches")
def historical_matches(
    competition_id: str = "CL",
    season: int = 2024,
    club_id: int | None = None,
    limit: int = 20,
    offset: int = 0,
):
    get_season(competition_id, season)
    if not 1 <= limit <= 100 or offset < 0:
        raise HTTPException(422, "limit: 1–100, offset >= 0")
    df = archive_games()
    group = df[(df.competition_id == competition_id) & (df.season == season)]
    if club_id is not None:
        group = group[(group.home_club_id == club_id) | (group.away_club_id == club_id)]
    group = group.sort_values(["date", "game_id"], ascending=False)
    return {
        "total": len(group),
        "matches": group.iloc[offset : offset + limit].to_dict("records"),
        "score_note": "Архивный счёт. Плей-офф может включать дополнительное время и серию пенальти; это не счёт за 90 минут. Модель обучается только на чемпионатах и групповых этапах.",
    }


@lru_cache(maxsize=1)
def archive_games():
    return pd.concat(
        [pd.read_csv(p) for p in sorted((ROOT / "data/matches").glob("*.csv"))], ignore_index=True
    )


@lru_cache(maxsize=13)
def valuation_snapshot(season):
    path = ROOT / "data/valuations" / f"{season}.csv"
    return pd.read_csv(path).fillna("").to_dict("records") if path.exists() else []


@app.get("/players/valuable")
def valuable_players(club_id: int, season: int = 2024, limit: int = 10):
    if not 2012 <= season <= 2024 or not 1 <= limit <= 10:
        raise HTTPException(422, "Сезон 2012–2024, limit 1–10")
    known = {c["id"] for c in catalog()["clubs"]}
    if club_id not in known:
        raise HTTPException(404, "Клуб отсутствует в архиве")
    rows = [
        r for r in valuation_snapshot(season) if r["season"] == season and r["club_id"] == club_id
    ]
    return {
        "as_of": f"{season + 1}-06-30",
        "club_id": club_id,
        "players": rows[:limit],
        "note": "Оценочная стоимость в EUR, не цена трансфера и не рейтинг качества. Последняя оценка не позднее 30 июня, давностью до 365 дней. Клуб уточнён последующими трансферами до этой даты; неоднозначные переходы исключены. Пустой список означает недостаток данных. Это срез клуба на дату, не полный состав сезона; позиция справочная.",
    }


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home_club_id: int = Field(gt=0)
    away_club_id: int = Field(gt=0)
    date: date
    competition_id: str = "ES1"
    season: int | None = Field(default=None, ge=2012, le=2024)

    @model_validator(mode="after")
    def different(self):
        if self.competition_id not in COMPETITIONS:
            raise ValueError("Неизвестный турнир")
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
def teams(competition_id: str = "ES1", season: int = 2024):
    season_row = get_season(competition_id, season)
    df = archive_games()
    df = df[(df.competition_id == competition_id) & (df.season == season)]
    names = {}
    for row in df.itertuples():
        names[int(row.home_club_id)] = row.home_club_name
        names[int(row.away_club_id)] = row.away_club_name
    active = set(season_row["club_ids"])
    return {
        "teams": [
            {"id": i, "name": n, "latest_season": i in active}
            for i, n in sorted(names.items(), key=lambda t: t[1])
        ],
        "snapshot_through": str(pd.Timestamp(df.date.max()).date()),
        "scenario_date": str(pd.Timestamp(df.date.max()).date() + timedelta(days=1)),
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
        feats = [match_features(r) for r in request.matches]
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
                for r in frame[
                    [
                        c
                        for c in [
                            "home_club_id",
                            "away_club_id",
                            "date",
                            "competition_id",
                            "season",
                        ]
                        if c in frame
                    ]
                ].to_dict("records")
            ]
            feats = [match_features(r) for r in rows]
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
        if value["report"].get("schema_version") != 2 or value["report"]["features"] != FEATURES:
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
