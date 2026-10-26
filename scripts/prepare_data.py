import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with zipfile.ZipFile(args.archive) as archive:
    matches = [n for n in archive.namelist() if Path(n).name == "games.csv"]
    if len(matches) != 1:
        raise SystemExit("В архиве должен быть один games.csv")
    raw = archive.read(matches[0])
frame = pd.read_csv(io.BytesIO(raw))
frame = frame[(frame.competition_id == "ES1") & frame.season.between(2012, 2024)]
columns = [
    "game_id",
    "season",
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_name",
    "away_club_name",
    "home_club_goals",
    "away_club_goals",
]
frame = frame[columns].sort_values(["date", "game_id"])
assert frame.game_id.is_unique
path = root / "data/laliga.csv"
frame.to_csv(path, index=False)
manifest = {
    "source": "User-supplied archive.zip / games.csv",
    "upstream_license": "Not provided; check redistribution rights",
    "competition": "ES1",
    "seasons": [2012, 2024],
    "rows": len(frame),
    "barcelona_id": 131,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "original_sha256": hashlib.sha256(raw).hexdigest(),
}
(root / "data/manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Extracted {len(frame)} La Liga matches")
