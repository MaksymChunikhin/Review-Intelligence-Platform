# Review Intelligence Platform — Code and Notebook Style Guide

## 1. Scope

This guide applies to Python modules, scripts, tests, and Jupyter notebooks in
this repository. Its purpose is to keep experiments reproducible and make the
same core pipeline reusable across Amazon product categories.

The guiding rule is:

```text
notebooks explain experiments;
src/ contains reusable implementation;
tests/ verify contracts and behavior.
```

## 2. Language and Learning Style

The codebase must work both as a professional portfolio artifact and as a
learning environment for its owner.

- Use English for identifiers, filenames, paths, schemas, API fields, command
  names, log messages, and technical artifact names. This keeps the project
  compatible with Python tooling, cloud services, and international teams.
- Write explanatory notebook Markdown in two languages. Within every Markdown
  cell, place the **complete English block first** and the **complete Russian
  block second**. Do not alternate English and Russian paragraphs in a long
  cell. The Russian text should explain the meaning, not merely translate
  isolated words.
- Split bilingual headings in the same order. Use an English heading for the
  English block and then a Russian heading for the Russian block. Avoid a
  combined heading such as `Conclusion / Вывод` when the cell contains long
  explanations.
- Write non-obvious code comments in two languages: one English comment
  followed by its Russian explanation. Explain the reason, assumption,
  constraint, input/output meaning, or trade-off.
- Do not comment every line or translate obvious Python syntax. Detailed means
  conceptually complete, not noisy. Prefer a bilingual block comment before a
  logical operation to repeated comments inside that operation.
- Public docstrings remain English so IDE help and generated API documentation
  stay conventional. When a concept needs a beginner-level explanation, add
  bilingual Markdown in the notebook or a bilingual explanatory block near the
  relevant example.
- Architecture documents may use English as their canonical language, but
  educational notebook narratives and work handoffs must explain unfamiliar
  terms in Russian.
- Write Russian explanations in ordinary learner-friendly language. Do not use
  a literal translation, professional shorthand, or an English loanword when a
  clear everyday phrase communicates the same meaning.
- At the first occurrence of a necessary technical term, state the plain
  meaning first and put the technical name in parentheses. After that, use the
  shorter term only when its meaning is already clear in the current notebook.
- A heading must describe the action or question in plain language. Do not put
  an unexplained architecture term such as "contract", "artifact", "scope",
  "pipeline", or "unseen product" in a Russian heading.
- Prefer concrete descriptions over compressed jargon. For example:

  | Avoid in Russian | Prefer in Russian |
  |---|---|
  | `контракт данных` without explanation | `правила структуры и качества данных (data contract)` |
  | `контракт полного каталога` | `проверка структуры и качества полного каталога` |
  | `невиданные товары` | `товары, отзывы о которых не использовались при обучении` |
  | `pipeline` / `пайплайн` | `последовательность обработки` or `процесс обработки` |
  | `scope` | `выбранная область анализа` |
  | `artifact` | the concrete object: `файл`, `модель`, `таблица`, or `отчет` |
- Explain metric names by the question they answer. A reader should not need to
  know terms such as accuracy, macro F1, calibration, or bootstrap before
  opening the notebook.
- Use Amazon and project domain terms consistently. Prefer the names from the
  canonical schemas and architecture documents.

Example:

```python
# Use source-row position because the Amazon source has no stable review ID.
# Используем позицию строки в исходнике, потому что Amazon-датасет не содержит
# стабильного идентификатора отзыва.
review_id = build_review_id(source_file, source_row_number)
```

## 3. Python Code

### Formatting and naming

- Follow PEP 8 and format code with Black's default line length of 88.
- Use `snake_case` for modules, functions, variables, and notebook filenames.
- Use `PascalCase` for classes and Pydantic models.
- Use `UPPER_SNAKE_CASE` for true module-level constants.
- Give analytical values explicit names such as `negative_review_count`; avoid
  unclear names such as `x`, `tmp`, or `data2` outside a tiny local scope.
- Prefer small functions with one responsibility and explicit inputs and
  outputs.

### Imports

Keep imports at the top of a Python file in three groups, separated by one
blank line:

1. standard library;
2. third-party packages;
3. local project modules.

Within each group, sort imports alphabetically. Do not use wildcard imports.
Avoid importing inside functions unless a documented optional dependency or a
measured startup-cost issue requires it.

```python
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.data.review_sampling import sample_balanced_reviews
```

### Types and contracts

- Add type hints to public functions, classes, and important data structures.
- Use Pydantic models for external and cross-layer contracts.
- Validate required columns, value ranges, dates, identifiers, and nullability
  at ingestion boundaries.
- Return structured values rather than unrelated positional tuples when the
  result has several meanings.
- Keep domain/category-specific behavior in configuration, taxonomy data, or
  model artifacts—not in duplicated branches of core pipeline code.

### Functions and state

- Pass paths, configuration, model versions, and random seeds explicitly.
- Use `pathlib.Path` for filesystem paths.
- Avoid hidden mutable global state and notebook-only dependencies.
- Make deterministic operations idempotent where practical.
- For large datasets, use streaming, chunking, batching, or bounded sampling.
  Loading a complete multi-million-row dataset into memory must be justified.
- Keep deterministic calculations in Python. Do not delegate counts,
  percentages, aggregations, or comparisons to an LLM.

### Comments and docstrings

- Comments explain **why**, an assumption, a non-obvious constraint, or a
  trade-off. Do not restate the next line of code.
- For meaningful explanatory comments, write English first and Russian second,
  following the language policy in Section 2.
- Give a beginner enough context to understand the decision, while keeping the
  comment current and directly above the code it explains.
- Do not leave commented-out code; version control already preserves history.
- Use `TODO(owner-or-issue): action and reason` for actionable unfinished work.
- Add a module docstring when a module has a clear responsibility.
- Add docstrings to public functions, classes, and methods. Start with a short
  imperative summary and document parameters, return values, exceptions, and
  important side effects when they are not obvious from the signature.

Good:

```python
# Use a bounded reservoir because the source file does not fit in memory.
# Используем выборку ограниченного размера, потому что исходный файл не
# помещается в оперативную память.
sample = sample_reviews(source_path, sample_size=100_000, random_state=42)
```

Avoid:

```python
# Sample the reviews.
sample = sample_reviews(source_path, sample_size=100_000, random_state=42)
```

### Errors, logging, and secrets

- Raise specific exceptions with enough context to locate the bad dataset,
  field, record, or artifact.
- Do not silently catch errors or continue with partially valid analytical
  results.
- Use logging in reusable pipelines. Reserve `print` for CLI output and concise
  notebook results.
- Never commit credentials, tokens, private URLs, or environment-specific
  secrets. Read them from environment variables or the deployment secret
  manager.

### Reproducibility and artifacts

- Define random seeds explicitly and reuse the same seed for comparable runs.
- Treat `data/raw/` as immutable input.
- Write generated data to a documented processed/artifact location.
- Record the dataset version, input window, schema version, model version,
  configuration, seed, and evaluation metrics for model experiments.
- Validate saved artifacts after writing them. Prefer atomic writes for large
  or expensive outputs.

### Artifact lifecycle and cleanup

- Do not keep superseded datasets, notebook-generated samples, temporary model
  directories, or caches after their replacement has been verified.
- Before deleting a large artifact, confirm its exact path, its replacement,
  and that no accepted pipeline still depends on it.
- Preserve immutable raw sources, the registered current canonical dataset,
  versioned reports, and explicitly registered historical model candidates.
- A temporary artifact must have an owner and a cleanup point. Notebook
  scratch data belongs in a temporary directory and is removed in the same run.
- When a notebook is replaced, remove obsolete outputs and update documentation
  references in the same change.
- Never treat an unregistered sample or unexplained train/test file as durable
  project data.

## 4. Jupyter Notebooks

### Naming and responsibility

- Name notebooks with a two-digit stage and a lowercase descriptive name:
  `09_aspect_discovery.ipynb`.
- One notebook answers one main analytical or experimental question.
- A notebook may orchestrate reusable code but must not become the only home
  of reusable schemas, adapters, metrics, or pipeline logic. Move those parts
  to `src/` and test them in `tests/`.

### Required structure

Use this order unless the experiment has a documented reason to differ:

1. title and purpose;
2. objectives or research questions;
3. inputs, outputs, and important assumptions;
4. **one imports cell**;
5. configuration, paths, versions, and random seed;
6. input existence and schema validation;
7. analysis or experiment sections;
8. evaluation and sanity checks;
9. artifact persistence and validation;
10. conclusion, limitations, and next step.

The narrative parts of this structure are bilingual. Complete the English part
of a Markdown cell first, then provide the complete Russian explanation at the
level needed by a learner who is seeing the method for the first time. Never
alternate the two languages paragraph by paragraph in a long cell.

### One imports cell

- Every import used by a notebook must appear in one code cell near the top.
- Do not add imports in later cells, including imports used by only one
  experiment branch.
- Use the same standard-library, third-party, and local grouping as Python
  modules.
- Keep configuration and expensive initialization out of the imports cell.
- Do not use `pip install` in a normal notebook run. Dependencies belong in the
  project dependency file.

Example:

```python
# Standard library / Стандартная библиотека
from pathlib import Path

# Third-party packages / Сторонние библиотеки
import numpy as np
import pandas as pd

# Local project modules / Локальные модули проекта
from src.data.review_sampling import sample_balanced_reviews
```

### Markdown and code cells

- Introduce each major section with Markdown that states the question and why
  the step exists. Keep the whole English explanation together, followed by
  the whole Russian explanation.
- Prefer a short narrative followed by code and an interpreted result. Do not
  create a long sequence of unexplained code cells.
- Keep one logical operation per code cell.
- Use comments inside code cells only for implementation details that Markdown
  would not explain better.
- Do not define large classes or long reusable functions in notebooks.
- Avoid dependence on accidental execution order. A notebook must run from a
  fresh kernel from top to bottom.

### Outputs and visualizations

- Show summaries and representative samples, not unbounded raw tables or log
  streams.
- Every chart must have a descriptive title, axis labels, readable units, and
  a consistent visual style.
- State whether statistics are exact, sampled, or estimated.
- Interpret important tables and charts in Markdown; do not leave conclusions
  implicit.
- Before considering a notebook complete, restart the kernel and run all cells.
  The committed notebook must contain no traceback or stale output from a
  different configuration.

### Notebook completion criteria

A completed notebook must make clear:

- what question was answered;
- which data and versions were used;
- which method and seed were used;
- what was measured;
- what artifact was created;
- what limitations remain;
- how the result affects the next architecture stage.

## 5. Tests and Quality Checks

- Add unit tests for reusable transformations, schemas, adapters, analytical
  functions, and edge cases.
- Add small integration tests for boundaries between ingestion, ML inference,
  analytics, retrieval, and API layers.
- Tests must use small fixtures, not the full local dataset.
- Any fixed bug should receive a regression test when practical.
- Before merging a change, format, lint, test, and execute affected notebooks
  from a clean kernel.

## 6. Review Checklist

- [ ] Reusable logic lives in `src/`, not only in a notebook.
- [ ] Names and contracts use the canonical Amazon project terminology.
- [ ] Imports are grouped and sorted.
- [ ] Each notebook has exactly one imports cell near the top.
- [ ] Notebook explanations and meaningful comments are bilingual: English,
      then Russian.
- [ ] Comments explain reasons or constraints rather than obvious syntax.
- [ ] New terminology and non-obvious results are explained for a learner.
- [ ] Public code has useful types and docstrings.
- [ ] Large-data operations are memory-bounded.
- [ ] Seeds, dataset versions, and model versions are explicit.
- [ ] Outputs are validated and conclusions state limitations.
- [ ] Tests cover the changed reusable behavior.
