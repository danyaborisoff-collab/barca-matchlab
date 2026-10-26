import json
from datetime import UTC, datetime

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.artifacts import ROOT, fingerprint, model_dir, save
from ml.features import DATA, FEATURES, LABELS, build_features, load_games


def scores(model, X, y):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    probabilities = probs[:, [classes.index(c) for c in LABELS]]
    return {
        "roc_auc_ovr_macro": float(
            roc_auc_score(y, probabilities, labels=LABELS, multi_class="ovr", average="macro")
        ),
        "accuracy": float(accuracy_score(y, model.predict(X))),
        "log_loss": float(log_loss(y, probabilities, labels=LABELS)),
    }


def train():
    frame = build_features(load_games())
    train_rows = frame[frame.season <= 2020]
    validation = frame[frame.season.between(2021, 2022)]
    test = frame[frame.season.between(2023, 2024)]
    results, pipelines = {}, {}
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, C=0.3, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=160, max_depth=7, min_samples_leaf=15, random_state=42, n_jobs=1
        ),
    }
    for name, estimator in candidates.items():
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=10)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        pipe.fit(train_rows[FEATURES], train_rows.result)
        results[name] = scores(pipe, validation[FEATURES], validation.result)
        pipelines[name] = pipe
    winner = max(results, key=lambda n: results[n]["roc_auc_ovr_macro"])
    champion = pipelines[winner]
    fit_rows = frame[frame.season <= 2022]
    champion.fit(fit_rows[FEATURES], fit_rows.result)
    dummy = DummyClassifier(strategy="prior").fit(fit_rows[FEATURES], fit_rows.result)
    selected = np.array(FEATURES)[champion.named_steps["select"].get_support()].tolist()
    report = {
        "selected_model": winner,
        "selection_basis": "Validation ROC-AUC, seasons 2021/22–2022/23; test not used for selection",
        "validation": results,
        "test": scores(champion, test[FEATURES], test.result),
        "baseline_test": scores(dummy, test[FEATURES], test.result),
        "split_rows": {"train": len(train_rows), "validation": len(validation), "test": len(test)},
        "refit_rows": len(fit_rows),
        "trained_through": str(fit_rows.date.max().date()),
        "test_from": str(test.date.min().date()),
        "test_to": str(test.date.max().date()),
        "features": FEATURES,
        "selected_features": selected,
        "dataset_sha256": fingerprint(DATA.read_bytes()),
        "created_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "classes": LABELS,
        "football_scope": "ES1 / seasons 2012–2024",
        "limitations": [
            "Historical snapshot, not live team strength",
            "No injury/lineup/odds features",
            "ROC-AUC is not profit or calibration",
            "Data origin/license not confirmed by user",
        ],
    }
    report["barcelona_test"] = scores(
        champion,
        test[(test.home_club_id == 131) | (test.away_club_id == 131)][FEATURES],
        test[(test.home_club_id == 131) | (test.away_club_id == 131)].result,
    )
    bundle = {"format": 1, "pipeline": champion, "report": report}
    save(bundle)
    (model_dir() / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "docs/metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test[["date", "home_club_id", "away_club_id", "result"]].head(12).to_csv(
        ROOT / "data/example_raw.csv", index=False
    )
    test[[*FEATURES, "result"]].head(12).to_csv(ROOT / "data/example_features.csv", index=False)
    print(
        json.dumps(
            {
                "selected_model": winner,
                "validation": results,
                "test": report["test"],
                "rows": report["split_rows"],
            },
            indent=2,
        )
    )
    return bundle


if __name__ == "__main__":
    train()
