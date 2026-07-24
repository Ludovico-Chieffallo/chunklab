from chunklab.chunkers.fixed import FixedChunker
from chunklab.chunkers.recursive import RecursiveChunker
from chunklab.chunkers.semantic import SemanticChunker
from chunklab.chunkers.structure import StructureChunker


def assert_spans_reconstruct(chunks, document):
    for c in chunks:
        s, e = c.char_span
        assert document.text[s:e] == c.text


def test_fixed_chunker(handbook):
    chunks = FixedChunker(chunk_size=200, overlap=32).chunk(handbook)
    assert len(chunks) > 3
    assert_spans_reconstruct(chunks, handbook)
    assert all(c.token_count <= 200 + 2 for c in chunks)


def test_recursive_chunker(handbook):
    chunks = RecursiveChunker(chunk_size=200, overlap=32).chunk(handbook)
    assert len(chunks) > 3
    assert_spans_reconstruct(chunks, handbook)


def test_structure_chunker_keeps_sections(handbook):
    chunks = StructureChunker(max_tokens=800).chunk(handbook)
    assert_spans_reconstruct(chunks, handbook)
    # The overtime sentence must be intact in exactly one section chunk.
    hits = [c for c in chunks if "Overtime is paid at 1.5x the regular hourly rate" in c.text]
    assert len(hits) == 1
    assert hits[0].section_path  # heading trail populated


def test_structure_chunker_table_intact(handbook):
    chunks = StructureChunker(max_tokens=800).chunk(handbook)
    table_chunks = [c for c in chunks if c.contains_table]
    assert table_chunks
    assert any("Pension match" in c.text and "Learning budget" in c.text for c in table_chunks)


def test_structure_merges_heading_only_sections(handbook):
    from chunklab.loaders.markdown_elements import extract_markdown_elements
    from chunklab.models import Document

    text = (
        "## Datasets\n\n### Upload\n\nBody text about uploads.\n\n"
        "### Query\n\nBody about querying.\n"
    )
    doc = Document(id="t", source_path="t.md", text=text, elements=extract_markdown_elements(text))
    chunks = StructureChunker(max_tokens=800).chunk(doc)
    # The bare '## Datasets' heading must not become its own tiny chunk.
    assert all(c.token_count > 5 for c in chunks)
    assert chunks[0].text.startswith("## Datasets")
    assert "### Upload" in chunks[0].text


def test_semantic_floor_vs_no_floor(handbook, fake_embedder):
    floored = SemanticChunker(fake_embedder, min_tokens=200, max_tokens=1000).chunk(handbook)
    no_floor = SemanticChunker(
        fake_embedder,
        min_tokens=0,
        max_tokens=1000,
        strategy_name="semantic_no_floor",
        breakpoint_percentile=40,
    ).chunk(handbook)
    assert_spans_reconstruct(floored, handbook)
    assert_spans_reconstruct(no_floor, handbook)
    # Floor: no chunk under 200 tokens (except a possibly-unmergeable single chunk)
    if len(floored) > 1:
        assert all(c.token_count >= 200 for c in floored[:-1] + floored[-1:]) or all(
            c.token_count >= 200 for c in floored
        )
    # No-floor with an aggressive percentile produces many more, smaller chunks.
    assert len(no_floor) > len(floored)


def test_semantic_ceiling(handbook, fake_embedder):
    chunks = SemanticChunker(fake_embedder, min_tokens=0, max_tokens=150).chunk(handbook)
    # Single-sentence chunks may exceed the ceiling; multi-sentence ones must not.
    for c in chunks:
        assert c.token_count <= 150 or "." not in c.text.strip()[:-1]
