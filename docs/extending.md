# Adding your own chunking strategy

chunklab answers "which way of splitting my documents retrieves best". That question is
only worth asking if the splitter your pipeline actually uses is in the comparison — with
its own rules about tables, clause markers or page headers. Otherwise chunklab can only
rank the five strategies it happens to ship.

A chunker is anything with a `chunk(document) -> list[Chunk]` method. There is no base
class to inherit.

## The shortest useful example

```python
from chunklab.chunkers.base import build_chunk
from chunklab.plugins import register_chunker

class ClauseChunker:
    """Splits our contracts on numbered clause markers."""

    def __init__(self, min_chars: int = 200):
        self.min_chars = min_chars

    def chunk(self, document):
        chunks, start = [], 0
        for match in re.finditer(r"\n(?=\d+\.\d+\s)", document.text):
            end = match.start()
            if end - start >= self.min_chars:
                chunks.append(build_chunk(document, "clause", len(chunks), (start, end)))
                start = end
        chunks.append(build_chunk(document, "clause", len(chunks), (start, len(document.text))))
        return chunks

register_chunker("clause", ClauseChunker, "Splits on numbered clause markers")
```

`clause` is now an ordinary strategy name, usable anywhere a built-in one is:

```yaml
strategies:
  - name: recursive
    params: {chunk_size: 512, overlap: 64}
  - name: clause
    params: {min_chars: 300}
```

`build_chunk` is worth using rather than constructing `Chunk` yourself: it fills in the
token count, whether the span overlaps a table, and the heading trail — the fields the
diagnostics read. Give it character spans into `document.text` and everything else follows.

## What the factory receives

Your factory is called with the strategy's `params` as keyword arguments. It is given
`embedder=` **only if its signature asks for one**, so a splitter that needs no model does
not have to accept an argument it will not use:

```python
class MySemanticChunker:
    def __init__(self, embedder=None, threshold: float = 0.4): ...
```

The name you register under wins over any `name` attribute on the class — it is what ends
up in chunk ids and in the report, so what you configured is what you read.

## Publishing a chunker from a package

Installed packages are discovered automatically through an entry point, with no import
needed in user code:

```toml
[project.entry-points."chunklab.chunkers"]
clause = "my_package.chunkers:ClauseChunker"
```

A plugin that fails to import is skipped with a warning rather than taking the run down
with it.

Names are protected: registering over a built-in raises, and registering the same name
twice needs `replace=True`. A silent override would change what a config file means with
no sign of it in the report.

## Evaluating a LangChain or LlamaIndex splitter

chunklab deliberately ships **no framework adapters** — it is not a RAG framework, and an
adapter would age with someone else's API. The plugin interface is the adapter, and the
wrapper is short enough to keep in your own repo:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
from chunklab.chunkers.base import build_chunk
from chunklab.plugins import register_chunker

class LangChainSplitter:
    def __init__(self, **kwargs):
        self._splitter = MarkdownHeaderTextSplitter(**kwargs)

    def chunk(self, document):
        chunks, cursor = [], 0
        for piece in self._splitter.split_text(document.text):
            text = piece.page_content
            start = document.text.find(text, cursor)
            if start < 0:          # the splitter rewrote the text; skip rather than guess
                continue
            cursor = start + len(text)
            chunks.append(build_chunk(document, "langchain", len(chunks), (start, cursor)))
        return chunks

register_chunker("langchain", LangChainSplitter, "MarkdownHeaderTextSplitter")
```

One caveat that matters for correctness: chunklab scores by locating gold snippets inside
retrieved chunks, so a chunk has to be a **verbatim span of the source document**. A
splitter that rewrites text — normalising whitespace, prepending headers to every piece,
stripping markdown — produces chunks whose offsets no longer line up. Find the piece in
the original text (as above) and skip what you cannot locate, rather than reporting a span
you did not verify.

## Checking it works

```bash
chunklab strategies          # your strategy appears with its description
chunklab run --docs ./docs --questions questions.yaml --config config.yaml
```

If your strategy scores oddly, read `%tiny` and `boundary` first: they usually explain it.
A chunker whose spans are wrong tends to show `boundary` near zero, because the reported
spans do not end at sentence boundaries even when the text does.
