"""Minimal self-check for cross-paper entity resolution. Run: python test_resolve.py"""
import os
os.environ.setdefault("GROQ_API_KEY", "test-key-for-import")

from rag_pipeline import _entity_match, _norm_label


def test():
    assert _norm_label("Multi-Head Attention!") == "multi head attention"

    existing = [{"label": "Multi-Head Attention", "paper_ids": ["a"]}]
    # exact (after normalisation) and fuzzy variants resolve to the same node
    assert _entity_match("multi-head attention", existing) is existing[0]
    assert _entity_match("Multi Head Attention", existing) is existing[0]
    # a genuinely different concept does not
    assert _entity_match("Positional Encoding", existing) is None
    # short/empty labels never false-match
    assert _entity_match("", existing) is None
    print("ok")


if __name__ == "__main__":
    test()
