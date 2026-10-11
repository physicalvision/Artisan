
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import bm25s

_NON_DISTINCTIVE_TAG_TOKENS = frozenset({"cadquery", "example", "examples"})
_NON_DISTINCTIVE_CODE_TOKENS = frozenset({"cq", "false", "none", "result", "self", "true"})
_STOPWORD_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "without",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def _tokenize(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(_normalize_text(value))


def _meaningful_tokens(value: str) -> list[str]:
    return [
        token
        for token in _tokenize(value)
        if token not in _NON_DISTINCTIVE_TAG_TOKENS and token not in _STOPWORD_TOKENS
    ]


def token_variants(token: str) -> list[str]:
    variants = [token]
    candidate: str | None = None
    if len(token) > 5 and token.endswith("ies"):
        candidate = f"{token[:-3]}y"
    elif len(token) > 6 and token.endswith("ing"):
        candidate = token[:-3]
    elif len(token) > 5 and token.endswith("ed"):
        candidate = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        candidate = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        candidate = token[:-1]

    if candidate and len(candidate) >= 4 and candidate not in variants:
        variants.append(candidate)

    if len(token) > 6 and token.endswith("e"):
        candidate = token[:-1]
        if len(candidate) >= 4 and candidate not in variants:
            variants.append(candidate)

    return variants


def split_identifier(identifier: str) -> list[str]:
    normalized = _normalize_text(identifier)
    if not normalized:
        return []

    expanded = _CAMEL_CASE_BOUNDARY_RE.sub(" ", identifier.replace("_", " ").replace("-", " "))
    split_tokens = _tokenize(expanded)

    ordered: list[str] = []
    seen: set[str] = set()
    for token in [normalized, *split_tokens]:
        if (
            token
            and token not in seen
            and token not in _STOPWORD_TOKENS
            and token not in _NON_DISTINCTIVE_TAG_TOKENS
            and token not in _NON_DISTINCTIVE_CODE_TOKENS
        ):
            ordered.append(token)
            seen.add(token)
    return ordered


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for token in _meaningful_tokens(text):
        if token in _NON_DISTINCTIVE_CODE_TOKENS:
            continue
        out.extend(token_variants(token))
    return out


def identifier_tokens(identifier: str) -> list[str]:
    out: list[str] = []
    for part in split_identifier(identifier):
        out.extend(token_variants(part))
    return out


def weighted_doc(*fields: tuple[list[str], int]) -> list[str]:
    out: list[str] = []
    for tokens, weight in fields:
        out += tokens * weight
    return out


@dataclass(slots=True)
class BM25Index:
    retriever: bm25s.BM25
    n_docs: int


def build_bm25(doc_token_lists: list[list[str]]) -> BM25Index:
    retriever = bm25s.BM25()
    if doc_token_lists:
        retriever.index(doc_token_lists, show_progress=False)
    return BM25Index(retriever=retriever, n_docs=len(doc_token_lists))


def search_bm25(index: BM25Index, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
    if not query_tokens or index.n_docs == 0:
        return []
    results = index.retriever.retrieve(
        [query_tokens],
        k=min(k, index.n_docs),
        corpus=list(range(index.n_docs)),
        show_progress=False,
    )
    out: list[tuple[int, float]] = []
    for raw_idx, raw_score in zip(results.documents[0], results.scores[0]):
        score = float(raw_score)
        if score <= 0:
            continue
        out.append((int(raw_idx), score))
    return out
