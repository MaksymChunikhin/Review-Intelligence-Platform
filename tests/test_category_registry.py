"""Tests for the versioned Amazon category registry builder."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.analytics.category_registry import (
    CATEGORY_REGISTRY_SCHEMA,
    build_category_registry,
    category_path_key,
    normalize_category_label,
)


def test_normalize_category_label_preserves_matching_equivalence() -> None:
    assert normalize_category_label("  Hair   Care ") == "hair care"
    assert normalize_category_label("Ｔｏｏｌｓ") == "tools"
    assert normalize_category_label(None) is None
    assert category_path_key([" Beauty  Care ", "Ｈａｉｒ Care"]) == (
        "beauty care > hair care"
    )


def test_build_category_registry_reconciles_complete_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "metadata.jsonl"
    source_records = [
        {
            "parent_asin": "P1",
            "main_category": "All Beauty",
            "categories": [" Beauty & Personal Care ", "Hair   Care"],
        },
        {
            "parent_asin": "P2",
            "main_category": "All Beauty",
            "categories": [
                "Ｂｅａｕｔｙ ＆ Ｐｅｒｓｏｎａｌ Ｃａｒｅ",
                "hair care",
            ],
        },
        {
            "parent_asin": "P3",
            "main_category": "All Beauty",
            "categories": ["Beauty & Personal Care", "Skin Care"],
        },
        {"parent_asin": "", "main_category": None, "categories": []},
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in source_records),
        encoding="utf-8",
    )
    output = tmp_path / "category_registry.parquet"
    report_path = tmp_path / "category_report.json"

    report = build_category_registry(
        source,
        output,
        dataset_version="beauty_test_v1",
        dataset_category="Beauty_and_Personal_Care",
        registry_schema_version="amazon_category_registry_v1",
        report_path=report_path,
        memory_limit="512MB",
    )
    registry = pd.read_parquet(output)

    assert report.input_rows == 4
    assert report.distinct_parent_asin_count == 3
    assert report.missing_parent_asin_rows == 1
    assert report.missing_main_category_rows == 1
    assert report.missing_category_path_rows == 1
    assert report.registry_row_count == 2
    assert report.distinct_category_path_count == 2
    assert report.category_node_count == 3
    assert registry["product_record_count"].sum() == 3
    assert registry["category_path_id"].is_unique
    hair_path = registry.loc[
        registry["category_path_key"]
        == "beauty & personal care > hair care"
    ].iloc[0]
    assert hair_path["product_record_count"] == 2
    assert list(hair_path["category_path"]) == [
        "Beauty & Personal Care",
        "Hair Care",
    ]
    assert pq.ParquetFile(output).schema_arrow.equals(
        CATEGORY_REGISTRY_SCHEMA,
        check_metadata=False,
    )
    assert report_path.is_file()
    assert not (tmp_path / "category_registry.tmp.parquet").exists()
