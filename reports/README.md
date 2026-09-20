# Reports

The local research workspace keeps current decision-relevant artifacts and a
small number of referenced lineage artifacts unpacked:

- `aspects/` — approved Taxonomy V2 and final population/evaluation summaries;
- `manual_review/` — completed human decisions only;
- `model_evaluation/` — the selected DistilBERT V2 and final comparison reports;
- `niches/` and `data_quality/` — dataset lineage and scope manifests;
- `seller_demo/` — active API workspaces and downloadable Excel reports.

Most historical V1/V2/V3 experiments, smoke runs, readiness checks, draft
review files, and superseded seller reports were bundled into:

`archive/legacy_reports_2026-09-13.tar.gz`

Restore the historical snapshot from the project root with:

```bash
tar -xzf reports/archive/legacy_reports_2026-09-13.tar.gz
```

The public portfolio repository intentionally excludes those generated files,
complete review text, manual-review queues, and the historical archive. Only
the aggregate, text-free metrics under `portfolio/` are published.
