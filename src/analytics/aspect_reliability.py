"""Turn independent silver metrics into explicit aspect reliability tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICY_VERSION = "aspect_silver_reliability_v1"


def classify_aspect_reliability(
    evaluation_path: str | Path,
    taxonomy_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Classify aspect metrics without presenting silver as human gold."""
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    metrics = evaluation["per_aspect"]
    rows = []
    for aspect in taxonomy["aspects"]:
        aspect_id = aspect["aspect_id"]
        score = metrics.get(
            aspect_id,
            {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "silver_count": 0,
                "prediction_count": 0,
            },
        )
        silver_count = int(score["silver_count"])
        f1 = float(score["f1"])
        if silver_count >= 20 and f1 >= 0.70:
            tier = "strong"
        elif silver_count >= 10 and f1 >= 0.55:
            tier = "usable"
        elif silver_count >= 5 and f1 >= 0.35:
            tier = "directional"
        else:
            tier = "experimental"
        rows.append(
            {
                "aspect_id": aspect_id,
                "canonical_name": aspect["canonical_name"],
                "reliability_tier": tier,
                "silver_count": silver_count,
                "prediction_count": int(score["prediction_count"]),
                "precision": float(score["precision"]),
                "recall": float(score["recall"]),
                "f1": f1,
                "comparative_analytics_allowed": tier in {"strong", "usable"},
                "default_ui_visibility": tier != "experimental",
            }
        )
    tier_counts: dict[str, int] = {}
    for row in rows:
        tier = row["reliability_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    payload = {
        "policy_version": POLICY_VERSION,
        "taxonomy_version": taxonomy["taxonomy_version"],
        "reference_kind": evaluation["reference_kind"],
        "warning": (
            "Tiers measure agreement with Gemini silver labels, not human-gold "
            "accuracy. Rating-based attention remains the primary complaint signal."
        ),
        "tier_rules": {
            "strong": "silver_count >= 20 and F1 >= 0.70",
            "usable": "silver_count >= 10 and F1 >= 0.55",
            "directional": "silver_count >= 5 and F1 >= 0.35",
            "experimental": "all remaining aspects",
        },
        "tier_counts": dict(sorted(tier_counts.items())),
        "aspects": rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_path", type=Path)
    parser.add_argument("taxonomy_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    payload = classify_aspect_reliability(
        args.evaluation_path, args.taxonomy_path, args.output_path
    )
    print(json.dumps(payload["tier_counts"], indent=2))


if __name__ == "__main__":
    main()
