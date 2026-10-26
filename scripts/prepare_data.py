import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw-dir",
    type=Path,
    required=True,
    help="Directory containing extracted source CSVs; games.csv renamed football_games.csv",
)
args = parser.parse_args()
R = Path(__file__).resolve().parents[1]
D = R / "data"
raw = args.raw_dir
comps = {
    "ES1": ("La Liga", "Испания", "league"),
    "GB1": ("Premier League", "Англия", "league"),
    "FR1": ("Ligue 1", "Франция", "league"),
    "L1": ("Bundesliga", "Германия", "league"),
    "IT1": ("Serie A", "Италия", "league"),
    "CL": ("Champions League", "Европа", "cup"),
    "EL": ("Europa League", "Европа", "cup"),
    "UCOL": ("Conference League", "Европа", "cup"),
}
x = pd.read_csv(raw / "football_games.csv")
x = x[x.competition_id.isin(comps) & x.season.between(2012, 2024)].copy()
x["date"] = pd.to_datetime(x.date)
assert (
    x.game_id.is_unique and not x[["date", "home_club_goals", "away_club_goals"]].isna().any().any()
)
cols = [
    "game_id",
    "competition_id",
    "season",
    "round",
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_goals",
    "away_club_goals",
    "home_club_name",
    "away_club_name",
]
x = x[cols].sort_values(["date", "game_id"])
(D / "matches").mkdir(exist_ok=True)
for c, g in x.groupby("competition_id"):
    g.to_csv(D / "matches" / f"{c}.csv", index=False)
train = x[
    x.competition_id.isin(["ES1", "GB1", "FR1", "L1", "IT1"])
    | x["round"].str.contains("Group", case=False)
].copy()
(D / "training").mkdir(exist_ok=True)
for c, g in train.groupby("competition_id"):
    g.to_csv(D / "training" / f"{c}.csv", index=False)
names = {}
for r in x.itertuples():
    names[int(r.home_club_id)] = r.home_club_name
    names[int(r.away_club_id)] = r.away_club_name
seasons = []
for (c, s), g in x.groupby(["competition_id", "season"]):
    ids = sorted(set(g.home_club_id) | set(g.away_club_id))
    end = g.date.max().strftime("%Y-%m-%d")
    champion = None
    table = []
    if comps[c][2] == "league":
        tab = {i: dict(club_id=i, name=names[i], played=0, points=0, gf=0, ga=0) for i in ids}
        for r in g.itertuples():
            for team, gf, ga in [
                (r.home_club_id, r.home_club_goals, r.away_club_goals),
                (r.away_club_id, r.away_club_goals, r.home_club_goals),
            ]:
                t = tab[team]
                t["played"] += 1
                t["points"] += 3 if gf > ga else 1 if gf == ga else 0
                t["gf"] += gf
                t["ga"] += ga
        table = sorted(
            tab.values(), key=lambda t: (-t["points"], -(t["gf"] - t["ga"]), -t["gf"], t["name"])
        )
        champion = table[0]["club_id"]
    else:
        finals = g[g["round"].str.lower().eq("final")]
        if len(finals) == 1:
            f = finals.iloc[0]
            champion = (
                int(f.home_club_id if f.home_club_goals > f.away_club_goals else f.away_club_id)
                if f.home_club_goals != f.away_club_goals
                else None
            )
    seasons.append(
        dict(
            competition_id=c,
            season=int(s),
            label=f"{s}/{str(s + 1)[2:]}",
            matches=len(g),
            through=end,
            club_ids=ids,
            champion_id=champion,
            champion=names.get(champion),
            standings=table,
            method="Расчёт по очкам, разнице и голам; без дисциплинарных санкций и специальных тай-брейков"
            if comps[c][2] == "league"
            else "Победитель финала по архивному счёту, который может включать дополнительное время и пенальти",
        )
    )
(D / "catalog.json").write_text(
    json.dumps(
        dict(
            competitions=[dict(id=c, name=v[0], region=v[1], kind=v[2]) for c, v in comps.items()],
            seasons=seasons,
            clubs=[dict(id=i, name=n) for i, n in sorted(names.items())],
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
# A valuation record belongs to its club at valuation date; use latest global value, then transfer history.
v = pd.read_csv(
    raw / "player_valuations.csv",
    usecols=["player_id", "date", "market_value_in_eur", "current_club_id"],
)
v["date"] = pd.to_datetime(v.date)
t = pd.read_csv(
    raw / "transfers.csv", usecols=["player_id", "transfer_date", "from_club_id", "to_club_id"]
)
t["transfer_date"] = pd.to_datetime(t.transfer_date)
p = (
    pd.read_csv(raw / "players.csv", usecols=["player_id", "name", "position", "url"])
    .drop_duplicates("player_id")
    .set_index("player_id")
)
snapshots = []
ambiguous = 0
for season in range(2012, 2025):
    cutoff = pd.Timestamp(season + 1, 6, 30)
    selected = (
        v[v.date <= cutoff]
        .sort_values(["date", "player_id"])
        .drop_duplicates("player_id", keep="last")
        .copy()
    )
    selected = selected[(cutoff - selected.date).dt.days <= 365]
    tx = t[t.transfer_date <= cutoff]
    lastdate = tx.groupby("player_id").transfer_date.transform("max")
    last = tx[tx.transfer_date.eq(lastdate)]
    # Same-day chains are not reliably ordered in this archive: omit conflicting destination records.
    conflicts = last.groupby("player_id").to_club_id.nunique()
    bad = set(conflicts[conflicts > 1].index)
    ambiguous += len(bad)
    last = last[~last.player_id.isin(bad)].drop_duplicates("player_id").set_index("player_id")
    selected = selected[~selected.player_id.isin(bad)]
    for row in selected.itertuples():
        club = int(row.current_club_id)
        source = "valuation_record"
        if row.player_id in last.index:
            transfer = last.loc[row.player_id]
            if transfer.transfer_date >= row.date:
                club = int(transfer.to_club_id)
                source = "transfer_after_valuation"
        if club not in names or row.player_id not in p.index or row.market_value_in_eur <= 0:
            continue
        ident = p.loc[row.player_id]
        snapshots.append(
            dict(
                season=season,
                as_of=cutoff.strftime("%Y-%m-%d"),
                club_id=club,
                player_id=int(row.player_id),
                name=ident["name"],
                position=ident.position,
                value_eur=int(row.market_value_in_eur),
                valuation_date=row.date.strftime("%Y-%m-%d"),
                club_evidence=source,
                url=ident.url,
            )
        )
a = (
    pd.DataFrame(snapshots)
    .sort_values(
        ["season", "club_id", "value_eur", "player_id"], ascending=[True, True, False, True]
    )
    .groupby(["season", "club_id"])
    .head(10)
)
(D / "valuations").mkdir(exist_ok=True)
for s, g in a.groupby("season"):
    g.to_csv(D / "valuations" / f"{s}.csv", index=False)
manifest = dict(
    version="0.8",
    scope=list(comps),
    seasons=[2012, 2024],
    archive_matches=len(x),
    training_matches=len(train),
    training_policy="Domestic leagues and European group/league stages only; knockout excluded because scores may contain shootouts",
    counts=x.groupby("competition_id").size().to_dict(),
    training_counts=train.groupby("competition_id").size().to_dict(),
    player_snapshot_rows=len(a),
    valuation_cutoff="June 30 after season; last valuation at most 365 days old; transfers after valuation adjust membership; ambiguous same-day transfers omitted",
    ambiguous_transfer_player_seasons=ambiguous,
    source="User-provided archive.zip, games.csv / players.csv / player_valuations.csv / transfers.csv; original provider Transfermarkt, reuse licence not verified",
    raw_sha256={
        f: hashlib.sha256((raw / f).read_bytes()).hexdigest()
        for f in ["football_games.csv", "players.csv", "player_valuations.csv", "transfers.csv"]
    },
)
(D / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("Messi 2018", a[(a.player_id == 28003) & (a.season == 2018)].to_string(index=False))
