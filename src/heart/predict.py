"""Load the packaged model and score patients.

Used by the API and usable on its own:
    python -m heart.predict sample_request.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from heart import config


@dataclass
class HeartModel:
    pipeline: Any
    metadata: dict

    @classmethod
    def load(cls, model_path: Path = config.MODEL_PATH,
             metadata_path: Path = config.METADATA_PATH) -> "HeartModel":
        pipeline = joblib.load(model_path)
        meta_file = Path(metadata_path)
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return cls(pipeline=pipeline, metadata=metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict(self, records: list[dict]) -> list[dict]:
        # astype(float) turns None (optional fields) into NaN for the imputers.
        X = pd.DataFrame.from_records(records, columns=config.INPUT_FEATURES)
        X = X.astype(float)
        proba = self.pipeline.predict_proba(X)[:, 1]
        out = []
        for p in proba:
            pred = int(p >= self.threshold)
            out.append({
                "prediction": pred,
                "label": "heart_disease" if pred else "no_heart_disease",
                "probability_disease": round(float(p), 4),
                # Confidence = probability of the class actually predicted.
                "confidence": round(float(p if pred else 1 - p), 4),
            })
        return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score patients from a JSON file")
    parser.add_argument("input", type=Path, help="JSON object or list of objects")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text())
    records = payload if isinstance(payload, list) else [payload]
    model = HeartModel.load()
    print(json.dumps(model.predict(records), indent=2))


if __name__ == "__main__":
    main()
