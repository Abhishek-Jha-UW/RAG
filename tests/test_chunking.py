from ingestion import chunk_narrative, dedupe_chunks


def test_chunk_narrative_respects_window():
    text = "para one words here.\n\n" + ("word " * 400)
    chunks = chunk_narrative(
        text,
        "t.txt",
        sheet=None,
        page=None,
        kind="narrative",
        chunk_words=50,
        overlap=10,
    )
    assert len(chunks) >= 2
    for c in chunks:
        w = c["text"].split()
        assert len(w) <= 120


def test_dedupe_chunks():
    a = [{"text": "same", "source": "a"}]
    b = [{"text": "same", "source": "b"}]
    out = dedupe_chunks(a + b)
    assert len(out) == 1
