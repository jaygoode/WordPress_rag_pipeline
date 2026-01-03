import pytest
import html
import re
from agentic_rag.data import BaseIngestionPipeline
from agentic_rag.data.cleaning import clean_text

@pytest.mark.skip(reason="Provide ingestion/e2e tests for your pipeline.")
def test_ingestion_pipeline_contract() -> None:
    """Replace with a test that executes your ingestion pipeline end-to-end."""
    assert issubclass(
        BaseIngestionPipeline, object
    ), "placeholder to keep pytest discovering this module"

#pytest tests/test_ingest.py::test_clean_text_removes_html_and_normalizes_whitespace
def test_clean_text_removes_html_and_normalizes_whitespace():
    raw = """
    "In a shortcode context, is there any difference here?               array(             'slideshow' => '',         ),       and               array(             'slideshow' => NULL,         ),       Is there a best practice for that?"
    """

    cleaned = clean_text(raw)
    
    assert "  " not in cleaned
    assert "\n" not in cleaned
    assert "array(" in cleaned
    assert "'slideshow'" in cleaned
    assert "=>" in cleaned
    assert "NULL" in cleaned
    assert "Is there a best practice for that?" in cleaned
    assert cleaned == cleaned.strip()