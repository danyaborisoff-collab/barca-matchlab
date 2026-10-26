import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml.artifacts import dumps, loads
from ml.features import FEATURES, build_features, load_games, raw_features
from ml.train import train


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    import os

    previous = os.environ.get("MODEL_DIR")
    os.environ["MODEL_DIR"] = str(tmp_path_factory.mktemp("models"))
    value = train()
    yield value
    if previous is None:
        os.environ.pop("MODEL_DIR", None)
    else:
        os.environ["MODEL_DIR"] = previous


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    with TestClient(app) as c:
        app.state.bundle = trained
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_frontend(client):
    assert "MATCHLAB" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_missing_model(client):
    app.state.bundle = None
    assert (
        client.post(
            "/predict",
            json={"matches": [{"home_club_id": 131, "away_club_id": 418, "date": "2025-05-26"}]},
        ).status_code
        == 503
    )


def test_teams(client):
    data = client.get("/teams").json()
    assert data["barcelona_id"] == 131
    assert any(t["id"] == 131 and t["name"] == "FC Barcelona" for t in data["teams"])


def test_prediction(client):
    r = client.post(
        "/predict",
        json={"matches": [{"home_club_id": 131, "away_club_id": 418, "date": "2025-05-26"}]},
    )
    assert r.status_code == 200
    row = r.json()["predictions"][0]
    assert sum(row["probabilities"].values()) == pytest.approx(1)
    assert set(row["probabilities"]) == {"home", "draw", "away"}
    assert set(row["features"]) == set(FEATURES)
    assert row["in_training_period"] is False
    assert "X-Process-Time" in r.headers


@pytest.mark.parametrize("home,away", [(131, 131), (999999, 418), (131, -1)])
def test_invalid_teams(client, home, away):
    r = client.post(
        "/predict",
        json={"matches": [{"home_club_id": home, "away_club_id": away, "date": "2025-05-26"}]},
    )
    assert r.status_code == 422


def test_feature_prediction(client):
    f = raw_features(app.state.games, 131, 418, "2025-05-26")
    assert client.post("/predict", json={"features": [f]}).status_code == 200


def test_feature_schema(client):
    assert client.post("/predict", json={"features": [{"elo_diff": 0}]}).status_code == 422


def test_empty_request(client):
    assert client.post("/predict", json={}).status_code == 422


def test_metrics(client):
    m = client.get("/metrics").json()
    assert len(m["validation"]) == 2
    assert 0 <= m["test"]["roc_auc_ovr_macro"] <= 1
    assert m["trained_through"] < m["test_from"]
    assert m["split_rows"] == {"train": 3420, "validation": 760, "test": 760}
    assert len(m["selected_features"]) == 10


def test_csv_raw(client):
    payload = "date,home_club_id,away_club_id\n2025-05-26,131,418\n"
    r = client.post("/predict-from-csv", files={"file": ("x.csv", payload, "text/csv")})
    assert r.status_code == 200
    assert r.json()["roc_auc"] is None
    assert r.json()["dataset"][0]["home_club_id"] == 131


def test_csv_metrics(client):
    frame = build_features(app.state.games)
    sample = frame[frame.season == 2024].groupby("result").head(1)
    raw = sample[[*FEATURES, "result"]].to_csv(index=False)
    r = client.post("/predict-from-csv", files={"file": ("x.csv", raw, "text/csv")})
    assert r.status_code == 200
    assert 0 <= r.json()["roc_auc"] <= 1
    assert r.json()["rows"] == 3


def test_csv_single_class(client):
    raw = "date,home_club_id,away_club_id,result\n2025-05-26,131,418,home\n"
    r = client.post("/predict-from-csv", files={"file": ("x.csv", raw, "text/csv")})
    assert r.status_code == 200
    assert r.json()["roc_auc"] is None


@pytest.mark.parametrize(
    "raw", ["bad\n1\n", "", "date,home_club_id,away_club_id\ninvalid,131,418\n"]
)
def test_csv_invalid(client, raw):
    assert (
        client.post("/predict-from-csv", files={"file": ("x.csv", raw, "text/csv")}).status_code
        == 422
    )


def test_admin_required(client):
    assert client.post("/train").status_code == 401
    assert client.get("/model").status_code == 401


def test_upload_signed(client, trained):
    data = dumps(trained)
    r = client.post(
        "/upload-model", headers={"X-Admin-Token": "test-token"}, files={"file": ("m.pkl", data)}
    )
    assert r.status_code == 200


def test_upload_rejects_pickle(client):
    data = pickle.dumps({"not": "a trusted model"})
    r = client.post(
        "/upload-model", headers={"X-Admin-Token": "test-token"}, files={"file": ("m.pkl", data)}
    )
    assert r.status_code == 422


def test_tamper(trained):
    data = dumps(trained)
    with pytest.raises(ValueError, match="Подпись"):
        loads(data[:-1] + bytes([data[-1] ^ 1]))


def test_download(client):
    r = client.get("/model", headers={"X-Admin-Token": "test-token"})
    assert r.status_code == 200
    assert loads(r.content)["format"] == 1


def test_no_future_leakage():
    frame = load_games()
    first_day = frame.date.min()
    changed = frame.copy()
    changed.loc[changed.date >= first_day, "home_club_goals"] = 20
    a, b = build_features(frame), build_features(changed)
    pd.testing.assert_frame_equal(
        a[a.date == first_day][FEATURES], b[b.date == first_day][FEATURES]
    )


def test_raw_ignores_current_and_future_scores():
    frame = load_games()
    day = "2024-04-01"
    changed = frame.copy()
    changed.loc[changed.date >= day, "away_club_goals"] = 20
    assert raw_features(frame, 131, 418, day) == raw_features(changed, 131, 418, day)


def test_reject_missing_data(tmp_path):
    frame = load_games().head(3)
    frame.loc[0, "home_club_goals"] = np.nan
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_games(path)


def test_reject_duplicate_data(tmp_path):
    frame = load_games().head(1)
    path = tmp_path / "dup.csv"
    pd.concat([frame, frame]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_games(path)
