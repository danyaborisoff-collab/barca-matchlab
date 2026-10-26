from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_points5",
    "away_points5",
    "home_gf5",
    "away_gf5",
    "home_ga5",
    "away_ga5",
    "home_rest",
    "away_rest",
    "home_games",
    "away_games",
]
LABELS = ["away", "draw", "home"]
COMPETITIONS = ["ES1", "GB1", "FR1", "L1", "IT1", "CL", "EL", "UCOL"]
FEATURES = [*BASE_FEATURES, *[f"tournament_{c}" for c in COMPETITIONS]]
DATA = Path(__file__).resolve().parents[1] / "data/training"


def load_games(path=DATA):
    df = (
        pd.concat([pd.read_csv(p) for p in sorted(Path(path).glob("*.csv"))], ignore_index=True)
        if Path(path).is_dir()
        else pd.read_csv(path)
    )
    required = [
        "game_id",
        "season",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
    ]
    if df[required].isna().any().any() or not df.game_id.is_unique:
        raise ValueError("Пропуски или повтор game_id в исходных данных")
    df["date"] = pd.to_datetime(df.date, errors="raise")
    for field in ["home_club_goals", "away_club_goals"]:
        if not df[field].between(0, 30).all() or (df[field] % 1 != 0).any():
            raise ValueError("Некорректное число голов")
    if (df.home_club_id == df.away_club_id).any():
        raise ValueError("Команда не может играть сама с собой")
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


class State:
    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.form = defaultdict(lambda: deque(maxlen=5))
        self.last = {}
        self.games = defaultdict(int)

    def snapshot(self, home, away, day):
        def stats(team):
            form = self.form[team]
            vals = np.mean(form, axis=0) if form else [1.0, 1.2, 1.2]
            rest = min(30, (day - self.last[team]).days) if team in self.last else 7
            return [self.elo[team], *map(float, vals), float(max(0, rest)), float(self.games[team])]

        h, a = stats(home), stats(away)
        values = [
            h[0] - a[0],
            h[0],
            a[0],
            h[1],
            a[1],
            h[2],
            a[2],
            h[3],
            a[3],
            h[4],
            a[4],
            h[5],
            a[5],
        ]
        return dict(zip(BASE_FEATURES, values, strict=True))

    def update_day(self, rows):
        changes = []
        for r in rows:
            h, a, hg, ag = (
                int(r.home_club_id),
                int(r.away_club_id),
                int(r.home_club_goals),
                int(r.away_club_goals),
            )
            expected = 1 / (1 + 10 ** ((self.elo[a] - self.elo[h] - 65) / 400))
            actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            changes.append((h, a, 24 * (actual - expected), hg, ag, r.date))
        for h, a, delta, hg, ag, day in changes:
            self.elo[h] += delta
            self.elo[a] -= delta
            self.form[h].append((3 if hg > ag else 1 if hg == ag else 0, hg, ag))
            self.form[a].append((3 if ag > hg else 1 if ag == hg else 0, ag, hg))
            self.last[h] = self.last[a] = day
            self.games[h] += 1
            self.games[a] += 1


def build_features(df):
    state, out = State(), []
    for day, group in df.groupby("date", sort=True):
        rows = list(group.itertuples(index=False))
        for r in rows:
            out.append(
                {
                    **r._asdict(),
                    **state.snapshot(int(r.home_club_id), int(r.away_club_id), day),
                    **context(r.competition_id),
                    "result": "home"
                    if r.home_club_goals > r.away_club_goals
                    else "draw"
                    if r.home_club_goals == r.away_club_goals
                    else "away",
                }
            )
        state.update_day(rows)
    return pd.DataFrame(out)


def context(competition_id):
    if competition_id not in COMPETITIONS:
        raise ValueError("Неизвестный турнир")
    return {f"tournament_{c}": float(c == competition_id) for c in COMPETITIONS}


def raw_features(df, home, away, day, competition_id="ES1"):
    flags = context(competition_id)
    ids = set(df.home_club_id) | set(df.away_club_id)
    if home == away or home not in ids or away not in ids:
        raise ValueError("Выберите две разные команды из датасета")
    day = pd.Timestamp(date.fromisoformat(str(day)))
    state = State()
    for _, group in df[df.date < day].groupby("date", sort=True):
        state.update_day(list(group.itertuples(index=False)))
    return {**state.snapshot(home, away, day), **flags}
