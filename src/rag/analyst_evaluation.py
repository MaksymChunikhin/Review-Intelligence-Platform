"""Run a fixed, reproducible preflight and generation check for the AI analyst."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.api.service import SellerAnalyticsStore
from src.common.project import find_project_root
from src.llm.gemini_client import GeminiStructuredClient
from src.rag.query_understanding import (
    infer_evidence_sentiment,
    understand_evidence_query,
)
from src.schemas.analyst import load_analyst_config


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _compact_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Keep answer evidence without duplicating full review text in the report."""
    return {
        "answer": answer["answer_ru"],
        "findings": answer["findings"],
        "recommendations": answer["recommendations"],
        "limitations": answer["limitations"],
        "source_ids": [source["citation_id"] for source in answer["sources"]],
        "evidence_review_count": answer["evidence_review_count"],
        "metadata": answer["metadata"],
    }


def run_evaluation(
    project_root: Path,
    evaluation_path: Path,
    *,
    generate: bool,
    external_processing_consent: bool,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate routing/retrieval and optionally generate validated answers."""
    if generate and not external_processing_consent:
        raise ValueError("--generate requires --external-processing-consent")
    specification = _load_json(evaluation_path)
    store = SellerAnalyticsStore(project_root)
    client: GeminiStructuredClient | None = None
    if generate:
        analyst_config = load_analyst_config(
            project_root / "config/analyst/gemini_grounded_analyst_v1.json"
        )
        client = GeminiStructuredClient(analyst_config)

    results: list[dict[str, Any]] = []
    previous_cases = {
        str(item["case_id"]): item
        for item in (previous_report or {}).get("cases", [])
    }
    try:
        for case in specification["cases"]:
            previous = previous_cases.get(str(case["case_id"]))
            reusable_status = (
                "generated_and_validated" if generate else "preflight_passed"
            )
            if previous is not None and previous.get("status") == reusable_status:
                results.append(previous)
                continue
            resolved_aspect, evidence_query = understand_evidence_query(
                case["question"]
            )
            resolved_sentiment = infer_evidence_sentiment(case["question"])
            evidence_limit = int(
                case.get(
                    "maximum_evidence_reviews",
                    specification["maximum_evidence_reviews"],
                )
            )
            context = store.rag_context(
                case["workspace_version"],
                question=case["question"],
                evidence_query=evidence_query,
                aspect_id=resolved_aspect,
                sentiment=resolved_sentiment,
                scope=specification["scope"],
                limit=evidence_limit,
            )
            route_matches = (
                resolved_aspect == case["expected_aspect_id"]
                and resolved_sentiment == case["expected_sentiment"]
            )
            result: dict[str, Any] = {
                **case,
                "resolved_aspect_id": resolved_aspect,
                "resolved_sentiment": resolved_sentiment,
                "evidence_query": evidence_query,
                "evidence_count": len(context["citations"]),
                "route_matches_expectation": route_matches,
                "retrieval_has_evidence": bool(context["citations"]),
            }
            if case["support_level"] == "requires_comparison_analytics":
                result["status"] = "capability_gap"
                result["reason"] = (
                    "The question requires deterministic multi-product analytics, "
                    "not synthesis from eight reviews."
                )
            elif not route_matches or not context["citations"]:
                result["status"] = "preflight_failed"
            elif not generate:
                result["status"] = "preflight_passed"
            else:
                try:
                    answer = store.ask_analyst(
                        case["workspace_version"],
                        question=case["question"],
                        aspect_id=resolved_aspect,
                        evidence_query=evidence_query,
                        sentiment=resolved_sentiment,
                        scope=specification["scope"],
                        maximum_evidence_reviews=evidence_limit,
                        external_processing_consent=True,
                        analyst_client=client,
                    )
                    result["status"] = "generated_and_validated"
                    result["generated"] = _compact_answer(answer)
                except Exception as error:  # retain the rest of the fixed run
                    result["status"] = "generation_failed"
                    result["error_type"] = type(error).__name__
                    result["error"] = str(error)
            results.append(result)
    finally:
        if client is not None:
            client.close()

    statuses: dict[str, int] = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    return {
        "evaluation_version": specification["evaluation_version"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generated_answers": generate,
        "scope": specification["scope"],
        "case_count": len(results),
        "summary": {
            "status_counts": statuses,
            "routing_matches": sum(
                bool(item["route_matches_expectation"]) for item in results
            ),
            "retrieval_with_evidence": sum(
                bool(item["retrieval_has_evidence"]) for item in results
            ),
        },
        "limitations": [
            "Structural validation does not prove that every generated claim is semantically correct.",
            "Aspect product rankings use the full exact competitor niche; repurchase signals use the saved direct competitors.",
            "Aspect rankings inherit the diagnostic limitations of aspect extraction and aspect sentiment.",
            "The evaluation uses the historical 2021-01-01 through 2023-09-12 snapshot.",
        ],
        "cases": results,
    }


def main() -> None:
    """Run the fixed analyst evaluation and materialize one compact report."""
    root = find_project_root(Path(__file__).parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=root / "config/analyst/shampoo_and_conditioner_analyst_eval_v1.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root / "reports/analyst/shampoo_and_conditioner_analyst_eval_v1.json",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--external-processing-consent", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already successful cases from the existing output report.",
    )
    args = parser.parse_args()
    previous_report = (
        _load_json(args.output_path)
        if args.resume and args.output_path.is_file()
        else None
    )
    report = run_evaluation(
        root,
        args.evaluation_path,
        generate=args.generate,
        external_processing_consent=args.external_processing_consent,
        previous_report=previous_report,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_suffix(".tmp.json")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(args.output_path)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
