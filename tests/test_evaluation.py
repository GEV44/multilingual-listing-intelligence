from __future__ import annotations

import json
from pathlib import Path

import pytest

from listing_intel.evaluation import (
    evaluate_dedup,
    evaluate_extraction,
    evaluate_ranking,
    main,
    ndcg_at_k,
    prf,
    to_markdown,
)
from listing_intel.extract import RulesExtractor
from listing_intel.synth import generate_posts


def test_prf() -> None:
    r = prf(8, 2, 2)
    assert r["precision"] == 0.8 and r["recall"] == 0.8 and r["f1"] == pytest.approx(0.8)
    assert prf(0, 0, 0)["f1"] == 0.0


def test_ndcg() -> None:
    assert ndcg_at_k([3, 2, 1], [3, 2, 1, 0], 3) == pytest.approx(1.0)
    worse = ndcg_at_k([1, 2, 3], [3, 2, 1], 3)
    assert worse is not None and worse < 1.0
    assert ndcg_at_k([0, 0], [0, 0, 0], 2) is None


def test_extraction_counts_and_standard_beats_hard() -> None:
    res = evaluate_extraction(generate_posts(150, 1), RulesExtractor())
    assert res["tier:standard"]["__micro__"]["f1"] > res["tier:hard"]["__micro__"]["f1"]
    assert res["all"]["street"]["f1"] > 0.9
    assert {"all", "lang:ru", "lang:hy", "lang:en"} <= set(res)


def test_ranking_systems_ordering() -> None:
    res = evaluate_ranking(300, 60, seed=3)
    s = res["systems"]
    assert (
        s["hand-set weights, gold facts (upper bound)"]["ndcg@10"]
        >= s["hand-set weights"]["ndcg@10"]
    )
    assert s["hand-set weights"]["ndcg@10"] > s["random (after hard filters)"]["ndcg@10"]
    assert abs(sum(res["learned_weights"].values()) - 1) < 1e-9


def test_dedup_combined_beats_single_signals() -> None:
    res = evaluate_dedup(40, seed=5)
    assert res["combined"]["f1"] >= max(res["photo_hash_only"]["f1"], res["spec_only"]["f1"])


def test_cli_writes_outputs(tmp_path: Path) -> None:
    js, md = tmp_path / "r.json", tmp_path / "r.md"
    main(["--posts", "60", "--requests", "20", "--flats", "12", "--write", str(js),
          "--markdown", str(md)])  # fmt: skip
    data = json.loads(js.read_text(encoding="utf8"))
    assert {"extraction", "ranking", "dedup"} <= set(data)
    assert "nDCG@10" in md.read_text(encoding="utf8")
    assert "micro-avg" in to_markdown(data)
