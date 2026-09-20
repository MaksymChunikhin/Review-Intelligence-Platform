"""Guard the structure and bilingual narrative of accepted notebooks."""

import json
import re
from pathlib import Path


ACCEPTED_NOTEBOOKS = [
    Path("notebooks/01_data_foundation/01_beauty_source_audit.ipynb"),
    Path("notebooks/01_data_foundation/02_cleaning_validation.ipynb"),
    Path("notebooks/01_data_foundation/03_amazon_contract_validation.ipynb"),
    Path("notebooks/01_data_foundation/04_amazon_category_registry.ipynb"),
    Path("notebooks/01_data_foundation/05_catalog_and_scope_validation.ipynb"),
    Path("notebooks/02_sentiment/01_tfidf_baseline.ipynb"),
    Path("notebooks/02_sentiment/02_distilbert_training.ipynb"),
    Path("notebooks/02_sentiment/03_error_analysis.ipynb"),
    Path("notebooks/03_aspects/01_taxonomy_review.ipynb"),
    Path("notebooks/03_aspects/02_extraction_evaluation_sample.ipynb"),
]

STALE_PHRASES = {
    "бальзамы для волос",
    "reviews_universal_sample.parquet",
    "product_catalog_sample.parquet",
    "selected_reviews_hair_conditioners.parquet",
    "domain_detection.json",
    "product_id values",
}


def cell_source(cell: dict[str, object]) -> str:
    """Return one notebook cell source as plain text."""
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else str(source)


def narrative_block_language(block: str) -> str:
    """Classify a Markdown block while allowing English technical terms."""
    cyrillic_count = len(re.findall(r"[А-Яа-яЁё]", block))
    latin_count = len(re.findall(r"[A-Za-z]", block))
    if cyrillic_count == 0:
        return "english"
    if latin_count == 0 or cyrillic_count > latin_count * 0.15:
        return "russian"
    return "english"


def test_accepted_notebooks_have_bilingual_markdown_and_one_import_cell() -> None:
    for path in ACCEPTED_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        markdown_cells = [
            cell for cell in cells if cell["cell_type"] == "markdown"
        ]
        import_cells = [cell for cell in cells if cell.get("id") == "imports"]

        assert cells[0]["cell_type"] == "markdown", path
        assert cells[-1]["cell_type"] == "markdown", path
        assert len(import_cells) == 1, path

        opening = cell_source(cells[0])
        assert "Objectives" in opening, path
        assert "Цели" in opening, path

        closing = cell_source(cells[-1])
        assert re.search(r"Conclusion|Interpretation", closing), path
        assert re.search(r"Вывод|Интерпретация", closing), path

        for cell in markdown_cells:
            narrative = cell_source(cell)
            assert re.search(r"[A-Za-z]", narrative), (path, cell.get("id"))
            assert re.search(r"[А-Яа-яЁё]", narrative), (path, cell.get("id"))


def test_markdown_keeps_complete_english_block_before_russian_block() -> None:
    for path in ACCEPTED_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            narrative = cell_source(cell)
            blocks = re.split(r"\n\s*\n", narrative.strip())
            languages = [narrative_block_language(block) for block in blocks]
            first_russian = languages.index("russian")

            assert "english" not in languages[first_russian:], (
                path,
                cell.get("id"),
                languages,
            )
            assert not re.search(
                r"^#{1,6}\s+.+\s/\s.+[А-Яа-яЁё]",
                narrative,
                flags=re.MULTILINE,
            ), (path, cell.get("id"))


def test_accepted_notebook_comment_blocks_are_bilingual() -> None:
    for path in ACCEPTED_NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue

            comment_blocks: list[list[str]] = []
            current_block: list[str] = []
            for line in cell_source(cell).splitlines():
                if line.lstrip().startswith("#"):
                    current_block.append(line.strip())
                elif current_block:
                    comment_blocks.append(current_block)
                    current_block = []
            if current_block:
                comment_blocks.append(current_block)

            for block in comment_blocks:
                comment = " ".join(block)
                assert re.search(r"[A-Za-z]", comment), (path, cell.get("id"))
                assert re.search(r"[А-Яа-яЁё]", comment), (
                    path,
                    cell.get("id"),
                )


def test_accepted_notebooks_do_not_reference_obsolete_workflow() -> None:
    for path in ACCEPTED_NOTEBOOKS:
        notebook_text = path.read_text(encoding="utf-8").casefold()
        for phrase in STALE_PHRASES:
            assert phrase.casefold() not in notebook_text, (path, phrase)
