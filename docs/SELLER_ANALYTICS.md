# Seller Analytics

## Active scope

The application uses the historical 2021–2023
`shampoo_and_conditioner_v1` section: 358,925 reviews across 21,651 reviewed
parent products. Product search and workspace creation support the seven
resolved exact competitor niches. Unresolved generic products are quarantined.

For a selected ASIN, competitor selection stays inside its exact Amazon path,
requires a compatible title-level product format and audience, excludes the
same brand, and records both the requested and actual minimum-review threshold.
The current workspace pipeline never silently reuses a legacy Conditioner-only
report, and it prefers one competitor per brand before filling any remaining
slots.

## Interpretation rules

- Comparative strengths and weaknesses use only aspects whose reliability tier
  permits comparative analytics.
- Both seller and competitor sides require at least 20 aspect observations.
- A strength or weakness requires at least a 5 percentage-point difference;
  the interface does not force a fixed number of findings.
- Aspect leaders are read from precomputed rankings across every eligible
  product in the selected product's exact competitor niche.
- Repurchase text signals compare only the selected product and its saved
  direct competitors.
- Ratings and AI aspect sentiment are separate filters. A five-star review may
  still contain a negative sentence about one aspect.

Every AI answer is limited to eight public review texts, uses explicit user
consent, and exposes its `[R1]`–`[R8]` review citations in the interface.
Aggregate numbers are calculated locally; Gemini receives compact facts and
the selected public evidence, without ASINs, user IDs, or internal IDs.

The Excel workbook and Markdown report are English, spreadsheet-formula safe,
and registered with SHA-256 hashes. Data remain a historical snapshot rather
than live Amazon market intelligence.

## Run locally

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the dashboard and
`http://127.0.0.1:8000/docs` for the API contract. Docker packaging is
deliberately deferred.
