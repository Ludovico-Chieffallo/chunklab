from chunklab.chunkers.fixed import FixedChunker
from chunklab.chunkers.semantic import SemanticChunker
from chunklab.chunkers.structure import StructureChunker
from chunklab.diagnostics.chunk_health import compute_chunk_health


def test_no_floor_has_high_pct_tiny(handbook, fake_embedder):
    docs = {handbook.id: handbook}
    no_floor = SemanticChunker(
        fake_embedder,
        min_tokens=0,
        max_tokens=1000,
        breakpoint_percentile=40,
        strategy_name="semantic_no_floor",
    ).chunk(handbook)
    floored = SemanticChunker(fake_embedder, min_tokens=200, max_tokens=1000).chunk(handbook)

    h_no_floor = compute_chunk_health(no_floor, docs, min_floor_tokens=200)
    h_floored = compute_chunk_health(floored, docs, min_floor_tokens=200)
    assert h_no_floor.pct_tiny > h_floored.pct_tiny
    assert h_no_floor.pct_tiny > 0.3


def test_boundary_health_fixed_vs_structure(handbook):
    docs = {handbook.id: handbook}
    fixed = FixedChunker(chunk_size=200, overlap=32).chunk(handbook)
    structure = StructureChunker(max_tokens=800).chunk(handbook)
    h_fixed = compute_chunk_health(fixed, docs)
    h_structure = compute_chunk_health(structure, docs)
    assert h_structure.boundary_health > h_fixed.boundary_health


def test_table_integrity(handbook):
    docs = {handbook.id: handbook}
    structure = StructureChunker(max_tokens=800).chunk(handbook)
    h = compute_chunk_health(structure, docs)
    assert h.table_integrity == 1.0
