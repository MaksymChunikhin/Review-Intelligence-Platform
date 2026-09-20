"""Tests for resumable Gemini aspect-discovery orchestration."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.analytics.aspect_taxonomy import (
    aggregate_discovery_results,
    build_taxonomy_proposal,
    build_taxonomy_review_queue,
    materialize_approved_taxonomy,
    normalize_aspect_candidates,
)
from src.llm.aspect_discovery import (
    iter_discovery_batches,
    prepare_discovery_batches,
    run_discovery_batches,
    validate_batch_result,
)
from src.llm.gemini_client import GeminiResponseMetadata
from src.schemas.aspects import (
    AspectDiscoveryBatchResult,
    AspectDiscoveryConfig,
    AspectMention,
    AspectMerge,
    AspectNormalizationResult,
    ReviewAspectObservation,
    SilverReviewPrediction,
)


def test_silver_review_prediction_merges_duplicate_aspects() -> None:
    prediction = SilverReviewPrediction.model_validate(
        {
            "item_id": "item_0001",
            "aspects": [
                {
                    "aspect_id": "scent",
                    "sentiment": "positive",
                    "customer_phrases": ["smells good"],
                },
                {
                    "aspect_id": "scent",
                    "sentiment": "negative",
                    "customer_phrases": ["too strong"],
                },
            ],
            "no_supported_aspect": False,
        }
    )

    assert len(prediction.aspects) == 1
    assert prediction.aspects[0].sentiment == "mixed"
    assert prediction.aspects[0].customer_phrases == [
        "smells good",
        "too strong",
    ]


def make_config(*, batch_size: int = 2) -> AspectDiscoveryConfig:
    """Return a compact discovery configuration for tests."""
    return AspectDiscoveryConfig(
        discovery_version="conditioners_discovery_test_v1",
        proposed_taxonomy_version="conditioners_taxonomy_test_v1",
        niche_id="hair_conditioners",
        niche_version="hair_conditioners_v1",
        dataset_version="beauty_test_v1",
        sample_schema_version="aspect_discovery_sample_v2",
        backend="vertex_ai",
        model="gemini-test",
        location="eu",
        batch_size=batch_size,
        temperature=0,
        seed=42,
        thinking_level="LOW",
        max_output_tokens=1000,
        max_retries=1,
        retry_base_seconds=0,
        max_validation_repairs=1,
        maximum_ungrounded_mention_share=0.05,
        minimum_candidate_sample_reviews=2,
        minimum_candidate_products=2,
        maximum_normalization_candidates=10,
    )


def write_sample(path: Path) -> None:
    """Write a small stratified discovery sample."""
    pd.DataFrame(
        {
            "dataset_version": ["beauty_test_v1"] * 4,
            "niche_id": ["hair_conditioners"] * 4,
            "niche_version": ["hair_conditioners_v1"] * 4,
            "sample_schema_version": ["aspect_discovery_sample_v2"] * 4,
            "review_id": ["R1", "R2", "R3", "R4"],
            "parent_asin": ["P1", "P2", "P3", "P4"],
            "product_title": ["One", "Two", "Three", "Four"],
            "rating": [1.0, 3.0, 5.0, 4.0],
            "review_year": [2021, 2022, 2023, 2021],
            "sampling_weight": [10.0, 20.0, 30.0, 40.0],
            "sampling_weight_semantics": [
                "inverse_review_inclusion_probability"
            ]
            * 4,
            "population_stratum_count": [10, 20, 30, 40],
            "sample_stratum_count": [1, 1, 1, 1],
            "review_text": [
                "The fragrance is strong.",
                "I like the fragrance.",
                "The fragrance lasts all day.",
                "Packaging is easy to use.",
            ],
            "stratum_id": [
                "stratum_a",
                "stratum_b",
                "stratum_c",
                "stratum_d",
            ],
        }
    ).to_parquet(path, index=False)


class FakeDiscoveryClient:
    """Return grounded structured observations without network access."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        contents: str,
        system_instruction: str,
        response_model: type[Any],
    ) -> tuple[AspectDiscoveryBatchResult, GeminiResponseMetadata, str]:
        self.calls += 1
        batch_id = contents.split("batch ", 1)[1].split(".", 1)[0]
        reviews = json.loads(contents.split("JSON array:\n", 1)[1])
        observations = []
        for review in reviews:
            phrase = (
                "fragrance"
                if "fragrance" in review["review_text"]
                else "Packaging"
            )
            label = "fragrance" if phrase == "fragrance" else "packaging"
            observations.append(
                ReviewAspectObservation(
                    review_id=review["review_id"],
                    mentions=[
                        AspectMention(
                            aspect_label=label,
                            topic_label=label,
                            topic_type="attribute",
                            customer_phrase=phrase,
                        )
                    ],
                )
            )
        result = AspectDiscoveryBatchResult(
            batch_id=batch_id,
            observations=observations,
        )
        metadata = GeminiResponseMetadata(
            response_id=f"response-{self.calls}",
            model_version="gemini-test-001",
            usage_metadata={
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        )
        return result, metadata, result.model_dump_json()


class FakeNormalizationClient:
    """Merge every selected test candidate into its own canonical aspect."""

    def generate(
        self,
        *,
        contents: str,
        system_instruction: str,
        response_model: type[Any],
    ) -> tuple[AspectNormalizationResult, GeminiResponseMetadata, str]:
        candidates = json.loads(contents.split("JSON:\n", 1)[1])
        aspects = [
            AspectMerge(
                aspect_id=item["candidate_key"].replace(" ", "_"),
                canonical_name=item["display_label"],
                definition=f"Customer discussion of {item['display_label']}.",
                parent_group=(
                    "packaging"
                    if item["candidate_key"] == "packaging"
                    else "sensory"
                ),
                source_candidate_keys=[item["candidate_key"]],
            )
            for item in candidates
        ]
        result = AspectNormalizationResult(
            proposed_taxonomy_version="conditioners_taxonomy_test_v1",
            aspects=aspects,
            excluded_candidates=[],
        )
        metadata = GeminiResponseMetadata(
            response_id="normalization-1",
            model_version="gemini-test-001",
            usage_metadata={"total_token_count": 20},
        )
        return result, metadata, result.model_dump_json()


def test_prepare_and_run_discovery_batches_is_resumable(tmp_path: Path) -> None:
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    plan = tmp_path / "plan.jsonl"
    report = prepare_discovery_batches(make_config(), sample, plan)
    batches = list(iter_discovery_batches(plan))

    assert report.sample_review_count == 4
    assert report.batch_count == 2
    assert len(batches) == 2
    assert {review.review_id for review in batches[0].reviews} == {"R1", "R2"}

    client = FakeDiscoveryClient()
    output = tmp_path / "responses"
    run = run_discovery_batches(
        make_config(), plan, output, client=client
    )
    resumed = run_discovery_batches(
        make_config(), plan, output, client=client
    )

    assert run.completed_batch_count == 2
    assert run.completed_review_count == 4
    assert run.total_token_count == 30
    assert resumed.pending_batch_count == 0
    assert client.calls == 2


def test_completed_discovery_run_resumes_without_initializing_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully saved run must remain inspectable without SDK credentials."""
    config = make_config()
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    plan = tmp_path / "plan.jsonl"
    output = tmp_path / "responses"
    prepare_discovery_batches(config, sample, plan)
    run_discovery_batches(config, plan, output, client=FakeDiscoveryClient())

    def fail_initialization(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("completed resume must not initialize a client")

    monkeypatch.setattr(
        "src.llm.aspect_discovery.GeminiStructuredClient",
        fail_initialization,
    )
    resumed = run_discovery_batches(config, plan, output)

    assert resumed.pending_batch_count == 0
    assert resumed.completed_review_count == 4


def test_validate_batch_result_rejects_unquoted_evidence(tmp_path: Path) -> None:
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    plan = tmp_path / "plan.jsonl"
    prepare_discovery_batches(make_config(), sample, plan)
    batch = next(iter_discovery_batches(plan))
    result = AspectDiscoveryBatchResult(
        batch_id=batch.batch_id,
        observations=[
            ReviewAspectObservation(
                review_id=review.review_id,
                mentions=[
                    AspectMention(
                        aspect_label="fragrance",
                        topic_label="fragrance",
                        topic_type="attribute",
                        customer_phrase="not present in source",
                    )
                ],
            )
            for review in batch.reviews
        ],
    )

    with pytest.raises(ValueError, match="verbatim"):
        validate_batch_result(batch, result)


def test_aggregate_normalize_and_build_taxonomy_proposal(tmp_path: Path) -> None:
    config = make_config()
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    plan = tmp_path / "plan.jsonl"
    responses = tmp_path / "responses"
    prepare_discovery_batches(config, sample, plan)
    run_discovery_batches(
        config,
        plan,
        responses,
        client=FakeDiscoveryClient(),
    )
    observations = tmp_path / "observations.parquet"
    candidates = tmp_path / "candidates.json"
    report = aggregate_discovery_results(
        config,
        plan,
        responses,
        observations,
        candidates,
    )

    assert report.review_count == 4
    assert report.mention_count == 4
    assert report.distinct_candidate_count == 2
    assert report.normalization_candidate_count == 1
    candidate_payload = json.loads(candidates.read_text())
    fragrance = candidate_payload["candidates"][0]
    assert fragrance["candidate_key"] == "fragrance"
    assert fragrance["weighted_review_support"] == 60.0

    normalization = tmp_path / "normalization.json"
    normalize_aspect_candidates(
        config,
        candidates,
        normalization,
        client=FakeNormalizationClient(),
    )
    proposal_path = tmp_path / "proposal.json"
    proposal = build_taxonomy_proposal(
        config,
        observations,
        candidates,
        normalization,
        proposal_path,
    )

    assert proposal["status"] == "candidate"
    assert proposal["aspect_count"] == 1
    assert proposal["aspects"][0]["aspect_id"] == "fragrance"
    assert proposal["aspects"][0]["weighted_review_support"] == 60.0

    review_queue_path = tmp_path / "review_queue.csv"
    review_queue = build_taxonomy_review_queue(
        proposal_path,
        review_queue_path,
    )
    assert len(review_queue) == 1
    assert review_queue.iloc[0]["source_candidate_keys"] == "fragrance"
    assert review_queue.iloc[0]["review_decision"] == ""
    assert review_queue_path.exists()

    review_queue.loc[0, "review_decision"] = "approve"
    review_queue.to_csv(review_queue_path, index=False)
    reviewed_definition = tmp_path / "reviewed_definition.json"
    reviewed_definition.write_text(
        json.dumps(
            {
                "taxonomy_version": "conditioners_taxonomy_test_v1",
                "review_schema_version": "aspect_taxonomy_review_v1",
                "reviewed_on": "2026-08-16",
                "reviewed_by": "test_owner",
                "aspects": [
                    {
                        "aspect_id": "fragrance",
                        "canonical_name": "Fragrance",
                        "definition": "Customer discussion of fragrance.",
                        "parent_group": "sensory",
                        "source_candidate_keys": ["fragrance"],
                        "additional_aliases": ["smell"],
                    }
                ],
                "excluded_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    approved_path = tmp_path / "approved.json"

    review_queue.loc[0, "review_decision"] = "exclude"
    review_queue.loc[0, "review_notes"] = "Exclude in this negative test."
    review_queue.to_csv(review_queue_path, index=False)
    with pytest.raises(ValueError, match="excluded review decision"):
        materialize_approved_taxonomy(
            proposal_path,
            review_queue_path,
            reviewed_definition,
            observations,
            candidates,
            approved_path,
        )

    review_queue.loc[0, "review_decision"] = "approve"
    review_queue.loc[0, "review_notes"] = ""
    review_queue.to_csv(review_queue_path, index=False)
    review_queue.loc[0, "canonical_name"] = "Tampered label"
    review_queue.to_csv(review_queue_path, index=False)
    with pytest.raises(ValueError, match="proposal fields were modified"):
        materialize_approved_taxonomy(
            proposal_path,
            review_queue_path,
            reviewed_definition,
            observations,
            candidates,
            approved_path,
        )

    review_queue.loc[0, "canonical_name"] = proposal["aspects"][0][
        "canonical_name"
    ]
    review_queue.to_csv(review_queue_path, index=False)
    approved = materialize_approved_taxonomy(
        proposal_path,
        review_queue_path,
        reviewed_definition,
        observations,
        candidates,
        approved_path,
    )

    assert approved["status"] == "approved"
    assert approved["aspect_count"] == 1
    assert approved["assigned_candidate_count"] == 1
    assert approved["excluded_candidate_count"] == 0
    assert approved["aspects"][0]["aliases"] == ["Fragrance", "smell"]
    assert approved_path.exists()


def test_aggregate_rejects_plan_from_another_discovery_version(
    tmp_path: Path,
) -> None:
    config = make_config()
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    plan = tmp_path / "plan.jsonl"
    responses = tmp_path / "responses"
    prepare_discovery_batches(config, sample, plan)
    run_discovery_batches(config, plan, responses, client=FakeDiscoveryClient())

    mismatched = config.model_copy(
        update={"discovery_version": "conditioners_discovery_test_v2"}
    )
    with pytest.raises(ValueError, match="config identities differ"):
        aggregate_discovery_results(
            mismatched,
            plan,
            responses,
            tmp_path / "observations.parquet",
            tmp_path / "candidates.json",
        )


def test_prepare_rejects_stale_sample_identity_and_weights(
    tmp_path: Path,
) -> None:
    config = make_config()
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    frame = pd.read_parquet(sample)
    frame["sample_schema_version"] = "aspect_discovery_sample_v1"
    frame.to_parquet(sample, index=False)

    with pytest.raises(ValueError, match="sample_schema_version"):
        prepare_discovery_batches(config, sample, tmp_path / "plan.jsonl")

    write_sample(sample)
    frame = pd.read_parquet(sample)
    frame.loc[0, "sampling_weight"] = 999.0
    frame.to_parquet(sample, index=False)
    with pytest.raises(ValueError, match="stratum weights"):
        prepare_discovery_batches(config, sample, tmp_path / "plan.jsonl")


def test_normalization_rejects_candidates_from_another_dataset(
    tmp_path: Path,
) -> None:
    config = make_config()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "discovery_version": config.discovery_version,
                "dataset_version": "another_dataset_v1",
                "niche_version": config.niche_version,
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate artifact identity"):
        normalize_aspect_candidates(
            config,
            candidates,
            tmp_path / "normalization.json",
            client=FakeNormalizationClient(),
        )


def test_proposal_rejects_normalization_from_another_model(
    tmp_path: Path,
) -> None:
    config = make_config()
    sample = tmp_path / "sample.parquet"
    write_sample(sample)
    plan = tmp_path / "plan.jsonl"
    responses = tmp_path / "responses"
    prepare_discovery_batches(config, sample, plan)
    run_discovery_batches(config, plan, responses, client=FakeDiscoveryClient())
    observations = tmp_path / "observations.parquet"
    candidates = tmp_path / "candidates.json"
    aggregate_discovery_results(
        config,
        plan,
        responses,
        observations,
        candidates,
    )
    normalization = tmp_path / "normalization.json"
    normalize_aspect_candidates(
        config,
        candidates,
        normalization,
        client=FakeNormalizationClient(),
    )
    normalization_payload = json.loads(normalization.read_text())
    normalization_payload["requested_model"] = "another-model"
    normalization.write_text(json.dumps(normalization_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization artifact identity"):
        build_taxonomy_proposal(
            config,
            observations,
            candidates,
            normalization,
            tmp_path / "proposal.json",
        )


def test_taxonomy_review_csv_escapes_spreadsheet_formulas(
    tmp_path: Path,
) -> None:
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(
        json.dumps(
            {
                "taxonomy_version": "taxonomy_v1",
                "status": "candidate",
                "aspects": [
                    {
                        "aspect_id": "packaging",
                        "canonical_name": "Packaging",
                        "parent_group": "packaging",
                        "source_candidate_keys": ["packaging"],
                        "weighted_population_share": 0.1,
                        "sample_review_count": 1,
                        "product_count": 1,
                        "definition": "Package quality.",
                        "evidence": [
                            {
                                "customer_phrase": "=WEBSERVICE(\"https://example.test\")"
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "queue.csv"

    build_taxonomy_review_queue(proposal_path, output)
    saved = pd.read_csv(output, keep_default_na=False)

    assert saved.loc[0, "evidence_phrases"].startswith("'=")
