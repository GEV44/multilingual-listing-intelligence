"""Evaluation harness: field-level extraction P/R/F1, ranking nDCG@k, de-duplication pair F1.

Everything runs on the synthetic data in :mod:`listing_intel.synth`. Run it with::

    python -m listing_intel.evaluation --write docs/results.json --markdown docs/results.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .config import Settings
from .dedup import Entry, cluster, compare, hamming, phash, spec_similarity
from .extract import Extractor, RulesExtractor
from .learn import fit_weights
from .match import DEFAULT_WEIGHTS, Match, features, price_usd, rank, reject_reason
from .schema import EXTRACTED_FIELDS, Listing
from .synth import (
    SynthPost,
    SynthRequest,
    generate_dedup_set,
    generate_posts,
    generate_requests,
    true_relevance,
)

Counts = dict[str, int]


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}


def _same(field: str, gold: Any, pred: Any) -> bool:
    if field == "area_m2":
        return abs(float(gold) - float(pred)) <= 0.5
    return bool(gold == pred)


def field_counts(gold: Listing, pred: Listing) -> dict[str, Counts]:
    out: dict[str, Counts] = {}
    for f in EXTRACTED_FIELDS:
        g, p = getattr(gold, f), getattr(pred, f)
        c = {"tp": 0, "fp": 0, "fn": 0}
        if g is None and p is not None:
            c["fp"] = 1
        elif g is not None and p is None:
            c["fn"] = 1
        elif g is not None and p is not None:
            if _same(f, g, p):
                c["tp"] = 1
            else:
                c["fp"] = c["fn"] = 1
        out[f] = c
    return out


def evaluate_extraction(
    posts: Sequence[SynthPost], extractor: Extractor
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-field P/R/F1 overall and sliced by language and phrasing tier."""
    slices: dict[str, dict[str, Counts]] = defaultdict(
        lambda: {f: {"tp": 0, "fp": 0, "fn": 0} for f in EXTRACTED_FIELDS}
    )
    for post in posts:
        pred = extractor.extract(post.text)
        counts = field_counts(post.gold, pred)
        for key in ("all", f"lang:{post.lang}", f"tier:{post.tier}"):
            for f, c in counts.items():
                for k, v in c.items():
                    slices[key][f][k] += v
    result: dict[str, dict[str, dict[str, float]]] = {}
    for key, per_field in slices.items():
        rows = {f: prf(**c) for f, c in per_field.items()}
        micro = prf(
            sum(c["tp"] for c in per_field.values()),
            sum(c["fp"] for c in per_field.values()),
            sum(c["fn"] for c in per_field.values()),
        )
        macro_f1 = sum(r["f1"] for r in rows.values()) / len(rows)
        rows["__micro__"] = micro
        rows["__macro_f1__"] = {"f1": macro_f1}
        result[key] = rows
    return result


# --- ranking ----------------------------------------------------------------------------------


def ndcg_at_k(gains: Sequence[int], ideal_pool: Sequence[int], k: int) -> float | None:
    """nDCG@k with gain 2^g - 1. ``None`` when the request has no relevant item at all."""
    ideal = sorted(ideal_pool, reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return None
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))
    return float(dcg / idcg)


def evaluate_ranking(
    n_posts: int, n_requests: int, seed: int, settings: Settings | None = None
) -> dict[str, Any]:
    cfg = settings or Settings()
    rules = RulesExtractor()
    posts = generate_posts(n_posts, seed)
    extracted = [rules.extract(p.text) for p in posts]
    gold = [p.gold for p in posts]
    requests = generate_requests(n_requests, seed + 1, cfg.usd_per_unit)
    split = len(requests) // 2
    train, test = requests[:split], requests[split:]

    def grades(req: SynthRequest) -> list[int]:
        rng = random.Random(req.id)
        return [true_relevance(req, g, cfg.usd_per_unit, rng) for g in gold]

    grade_cache = {r.id: grades(r) for r in requests}

    # Training rows: strict-match candidates (extracted listings) labelled by true relevance.
    rows: list[tuple[dict[str, float], int]] = []
    for req in train:
        b_usd = cfg.to_usd(req.request.budget_max, req.request.currency)
        for lst, grade in zip(extracted, grade_cache[req.id], strict=True):
            usd = price_usd(lst, cfg)
            if reject_reason(lst, req.request, usd, b_usd) is None and usd is not None:
                rows.append((features(lst, req.request, usd, b_usd)[0], grade))
    learned = fit_weights(rows)

    def run(
        pool: Sequence[Listing], weights: dict[str, float] | None, shuffle: bool = False
    ) -> Callable[[SynthRequest], list[Match]]:
        def go(req: SynthRequest) -> list[Match]:
            big = req.request.model_copy(update={"top_k": 100})
            m = rank([(str(i), lst) for i, lst in enumerate(pool)], big, cfg, weights).matches
            if shuffle:
                random.Random(req.id).shuffle(m)
            return m

        return go

    systems: dict[str, Callable[[SynthRequest], list[Match]]] = {
        "random (after hard filters)": run(extracted, None, shuffle=True),
        "price only (after hard filters)": run(extracted, {"price": 1.0}),
        "hand-set weights": run(extracted, None),
        "learned weights": run(extracted, learned),
        "hand-set weights, gold facts (upper bound)": run(gold, None),
    }
    out: dict[str, Any] = {
        "systems": {},
        "learned_weights": learned,
        "default_weights": DEFAULT_WEIGHTS,
        "n_train_requests": len(train),
        "n_test_requests": len(test),
        "n_train_rows": len(rows),
    }
    for name, system in systems.items():
        n5: list[float] = []
        n10: list[float] = []
        for req in test:
            g = grade_cache[req.id]
            ranked = [g[int(m.id)] for m in system(req)]
            a, b = ndcg_at_k(ranked, g, 5), ndcg_at_k(ranked, g, 10)
            if a is not None and b is not None:
                n5.append(a)
                n10.append(b)
        out["systems"][name] = {
            "ndcg@5": sum(n5) / len(n5),
            "ndcg@10": sum(n10) / len(n10),
            "n_requests": len(n5),
        }
    return out


# --- de-duplication -----------------------------------------------------------------------------


def _pair_metrics(ids_flat: Sequence[int], decide: Callable[[int, int], bool]) -> dict[str, float]:
    tp = fp = fn = 0
    n = len(ids_flat)
    for i in range(n):
        for j in range(i + 1, n):
            same = ids_flat[i] == ids_flat[j]
            pred = decide(i, j)
            tp += same and pred
            fp += (not same) and pred
            fn += same and (not pred)
    return prf(tp, fp, fn)


def evaluate_dedup(n_flats: int, seed: int, settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or Settings()
    rules = RulesExtractor()
    items = generate_dedup_set(n_flats, seed)
    entries = [
        Entry(it.id, rules.extract(it.text), tuple(phash(p) for p in it.photos)) for it in items
    ]
    flats = [it.flat_id for it in items]

    def photo_only(i: int, j: int) -> bool:
        a, b = entries[i].photo_hashes, entries[j].photo_hashes
        return bool(a and b and min(hamming(x, y) for x in a for y in b) <= cfg.photo_hamming_max)

    def spec_only(i: int, j: int) -> bool:
        s = spec_similarity(entries[i].listing, entries[j].listing)
        return s is not None and s >= cfg.spec_only_threshold

    def combined(i: int, j: int) -> bool:
        return compare(entries[i], entries[j], cfg).is_duplicate

    clusters = cluster(entries, cfg)
    return {
        "n_items": len(items),
        "n_flats": len(set(flats)),
        "n_photoless_items": sum(1 for e in entries if not e.photo_hashes),
        "photo_hash_only": _pair_metrics(flats, photo_only),
        "spec_only": _pair_metrics(flats, spec_only),
        "combined": _pair_metrics(flats, combined),
        "n_clusters_found": len(clusters),
    }


# --- reporting ------------------------------------------------------------------------------------


def to_markdown(results: dict[str, Any]) -> str:
    ex = results["extraction"]
    lines = [
        "### Extraction (rules extractor, field-level)",
        "",
        "| field | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for f in EXTRACTED_FIELDS:
        r = ex["all"][f]
        lines.append(
            f"| {f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{int(r['support'])} |"
        )
    m = ex["all"]["__micro__"]
    lines.append(
        f"| **micro-avg** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
        f"{int(m['support'])} |"
    )
    lines += ["", "| slice | micro P | micro R | micro F1 | macro F1 |", "|---|---|---|---|---|"]
    for key in sorted(k for k in ex if k != "all"):
        mi = ex[key]["__micro__"]
        lines.append(
            f"| {key} | {mi['precision']:.3f} | {mi['recall']:.3f} | {mi['f1']:.3f} | "
            f"{ex[key]['__macro_f1__']['f1']:.3f} |"
        )
    rk = results["ranking"]
    lines += [
        "",
        f"### Ranking ({rk['n_test_requests']} held-out synthetic requests)",
        "",
        "| system | nDCG@5 | nDCG@10 |",
        "|---|---|---|",
    ]
    for name, s in rk["systems"].items():
        lines.append(f"| {name} | {s['ndcg@5']:.3f} | {s['ndcg@10']:.3f} |")
    dd = results["dedup"]
    lines += [
        "",
        f"### De-duplication ({dd['n_items']} posts of {dd['n_flats']} flats, pairwise)",
        "",
        "| method | precision | recall | F1 |",
        "|---|---|---|---|",
    ]
    for key, label in (
        ("photo_hash_only", "photo pHash only"),
        ("spec_only", "spec match only"),
        ("combined", "combined (default)"),
    ):
        r = dd[key]
        lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    return "\n".join(lines) + "\n"


def run_all(posts: int, requests: int, flats: int, seed: int) -> dict[str, Any]:
    return {
        "config": {"posts": posts, "requests": requests, "flats": flats, "seed": seed},
        "extraction": evaluate_extraction(generate_posts(posts, seed + 1000), RulesExtractor()),
        "ranking": evaluate_ranking(posts, requests, seed),
        "dedup": evaluate_dedup(flats, seed + 2000),
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the synthetic evaluation suite.")
    ap.add_argument("--posts", type=int, default=1000)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--flats", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--write", type=Path, help="write JSON results here")
    ap.add_argument("--markdown", type=Path, help="write the markdown tables here")
    args = ap.parse_args(argv)
    results = run_all(args.posts, args.requests, args.flats, args.seed)
    md = to_markdown(results)
    print(md)
    if args.write:
        args.write.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf8")
    if args.markdown:
        args.markdown.write_text(md, encoding="utf8")


if __name__ == "__main__":
    main()
