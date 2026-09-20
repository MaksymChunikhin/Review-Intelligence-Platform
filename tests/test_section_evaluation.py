"""Tests for disjoint section evaluation artifacts."""

from pathlib import Path

import pandas as pd

from src.data.section_evaluation import combine_section_evaluation_samples


def test_combine_section_evaluation_samples_keeps_components_disjoint(
    tmp_path: Path,
) -> None:
    common = {
        "parent_asin": ["p1", "p2"],
        "competitor_niche_id": ["a", "b"],
        "stratum_id": ["a|one", "b|one"],
        "sampling_weight": [2.0, 3.0],
    }
    representative = pd.DataFrame(
        {"review_id": ["r1", "r2"], **common}
    )
    targeted = pd.DataFrame(
        {
            "review_id": ["r3", "r4"],
            "parent_asin": ["p3", "p4"],
            "competitor_niche_id": ["a", "b"],
            "stratum_id": ["a|two", "b|two"],
            "sampling_weight": [4.0, 5.0],
        }
    )
    representative_path = tmp_path / "representative.parquet"
    targeted_path = tmp_path / "targeted.parquet"
    discovery_path = tmp_path / "discovery.parquet"
    output_path = tmp_path / "evaluation.parquet"
    representative.to_parquet(representative_path, index=False)
    targeted.to_parquet(targeted_path, index=False)
    pd.DataFrame({"review_id": ["r5"]}).to_parquet(
        discovery_path, index=False
    )

    report = combine_section_evaluation_samples(
        representative_path,
        targeted_path,
        discovery_path,
        output_path,
        evaluation_version="evaluation_v1",
    )
    output = pd.read_parquet(output_path)

    assert report.total_review_count == 4
    assert output["review_id"].is_unique
    assert set(output["sampling_component"]) == {"representative", "targeted"}
    assert output.loc[
        output["sampling_component"] == "targeted", "evaluation_weight"
    ].isna().all()
