"""Asymmetric query/passage prefixes required by some embedding models.

Several retrieval models are trained asymmetrically: the query and the passage
go through the same weights but with different instruction prefixes. E5 is the
strict case — `intfloat/*e5*` models are trained with literal `query: ` and
`passage: ` prefixes, and omitting them degrades retrieval badly while raising
no error at all. That silent cliff is exactly the kind of thing chunklab exists
to expose, so it must not have it itself.

Prefixes are applied inside the embedder, so every caller (retriever, semantic
chunker, cache) gets them right without knowing they exist.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixScheme:
    query: str = ""
    passage: str = ""

    def __bool__(self) -> bool:
        return bool(self.query or self.passage)


NONE = PrefixScheme()

#: E5 family: prefixes are mandatory, both sides.
E5 = PrefixScheme(query="query: ", passage="passage: ")

#: BGE family: the passage is embedded bare; the query carries a retrieval
#: instruction. BGE's own model card recommends it for short queries.
BGE_EN = PrefixScheme(query="Represent this sentence for searching relevant passages: ")

BGE_ZH = PrefixScheme(query="为这个句子生成表示以用于检索相关文章：")


def scheme_for(model_name: str) -> PrefixScheme:
    """The prefix scheme a model was trained with, or NONE when unknown.

    Unknown models get no prefix: inventing one would be worse than omitting it.
    """
    name = model_name.lower()
    if "e5" in name:
        # e5-*-unsupervised and the `-instruct` variants use other conventions;
        # only claim the plain supervised checkpoints.
        if "instruct" in name:
            return NONE
        return E5
    if "bge" in name:
        if "-zh" in name:
            return BGE_ZH
        if "bge-m3" in name:
            return NONE  # m3 is trained without an instruction prefix
        return BGE_EN
    return NONE
