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
    assert sum(m["split_rows"].values()) == 26902
    assert len(m["per_competition_test"]) == 8
    assert len(m["selected_features"]) == 18


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


def test_competition_catalog(client):
    result = client.get("/competitions").json()
    assert {c["id"] for c in result["competitions"]} == {
        "ES1",
        "GB1",
        "FR1",
        "L1",
        "IT1",
        "CL",
        "EL",
        "UCOL",
    }
    assert result["seasons"] == list(range(2024, 2011, -1))


@pytest.mark.parametrize("competition", ["ES1", "GB1", "FR1", "L1", "IT1", "CL", "EL", "UCOL"])
def test_season_and_awards(client, competition):
    r = client.get("/season-summary", params={"competition_id": competition, "season": 2024})
    assert r.status_code == 200
    assert r.json()["champion_id"]
    assert len(r.json()["awards"]) == 1
    assert r.json()["awards"][0]["source"].startswith("https://")


def test_historical_award_missing_is_explicit(client):
    r = client.get("/season-summary?competition_id=ES1&season=2012").json()
    assert r["awards"] == []
    assert "Отсутствие" in r["award_note"]


def test_unknown_competition(client):
    assert client.get("/teams?competition_id=INVALID").status_code == 404
    assert client.get("/season-summary?competition_id=UCOL&season=2012").status_code == 404


def test_european_cross_league_prediction(client):
    r = client.post(
        "/predict",
        json={
            "matches": [
                {
                    "home_club_id": 131,
                    "away_club_id": 27,
                    "date": "2025-05-26",
                    "competition_id": "CL",
                    "season": 2024,
                }
            ]
        },
    )
    assert r.status_code == 200
    f = r.json()["predictions"][0]["features"]
    assert f["tournament_CL"] == 1
    assert f["tournament_ES1"] == 0


def test_wrong_domestic_membership(client):
    r = client.post(
        "/predict",
        json={
            "matches": [
                {
                    "home_club_id": 131,
                    "away_club_id": 27,
                    "date": "2025-05-26",
                    "competition_id": "ES1",
                }
            ]
        },
    )
    assert r.status_code == 422


def test_scenario_outside_selected_season(client):
    r = client.post(
        "/predict",
        json={
            "matches": [
                {
                    "home_club_id": 131,
                    "away_club_id": 418,
                    "date": "2025-05-26",
                    "competition_id": "ES1",
                    "season": 2012,
                }
            ]
        },
    )
    assert r.status_code == 422


def test_fixtures_and_pagination(client):
    a = client.get("/matches?competition_id=CL&season=2024&limit=1").json()
    b = client.get("/matches?competition_id=CL&season=2024&limit=1&offset=1").json()
    assert a["total"] == 189
    assert a["matches"][0]["game_id"] != b["matches"][0]["game_id"]
    assert a["matches"][0]["round"] == "Final"
    assert client.get("/matches?limit=101").status_code == 422


def test_fixture_club_filter(client):
    a = client.get("/matches?competition_id=CL&season=2024&club_id=131").json()
    assert a["matches"] and all(131 in [m["home_club_id"], m["away_club_id"]] for m in a["matches"])


def test_penalty_scores_not_training_labels():
    df = load_games()
    assert not df.game_id.isin([2690027, 3563783, 3813154]).any()
    assert (
        df[df.competition_id.isin(["CL", "EL", "UCOL"])]["round"]
        .str.contains("Group", case=False)
        .all()
    )


def test_historical_valuation_messi(client):
    rows = client.get("/players/valuable?club_id=131&season=2018").json()["players"]
    messi = next(p for p in rows if p["player_id"] == 28003)
    assert messi["value_eur"] == 150000000
    assert messi["valuation_date"] == "2019-06-11"
    assert all(p["valuation_date"] <= p["as_of"] == "2019-06-30" for p in rows)
    assert not any(
        p["player_id"] == 28003
        for p in client.get("/players/valuable?club_id=131&season=2022").json()["players"]
    )


def test_valuation_validation(client):
    assert client.get("/players/valuable?club_id=999999").status_code == 404
    assert client.get("/players/valuable?club_id=131&season=2030").status_code == 422
    assert client.get("/players/valuable?club_id=131&limit=11").status_code == 422


def test_all_valuation_snapshots_no_future():
    from pathlib import Path

    for p in (Path(__file__).resolve().parents[1] / "data/valuations").glob("*.csv"):
        frame = pd.read_csv(p)
        age = (pd.to_datetime(frame.as_of) - pd.to_datetime(frame.valuation_date)).dt.days
        assert age.between(0, 365).all()
        assert not frame.duplicated(["season", "player_id"]).any()
        assert frame.groupby("club_id").size().max() <= 10
