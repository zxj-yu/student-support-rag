"""Week 1 starter tests. Expanded in Week 2 with retrieval + API coverage."""
from app.services.retrieval import chunk_text


def test_chunk_text_splits_long_input():
    # two paragraphs that exceed the limit should produce 2 chunks
    text = ("a" * 400) + "\n\n" + ("b" * 400)
    chunks = chunk_text(text, max_chars=600)
    assert len(chunks) == 2


def test_chunk_text_keeps_short_input_together():
    text = "short para one\n\nshort para two"
    chunks = chunk_text(text, max_chars=600)
    assert len(chunks) == 1


def test_chunk_text_never_empty():
    assert chunk_text("") == [""]
