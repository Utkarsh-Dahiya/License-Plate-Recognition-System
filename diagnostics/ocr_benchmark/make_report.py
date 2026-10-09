# OCR BENCHMARK REPORT GENERATOR — reads results.json, writes
# BENCHMARK_REPORT.md and per_sample_results.json. Read-only.
from __future__ import annotations

import json
import statistics
from pathlib import Path

import benchmark_core as bc

RES = bc.OUT_DIR / "results.json"


def pct(x):
    return f"{100.0 * x:.1f}%"


def load():
    return json.loads(RES.read_text(encoding="utf-8"))


def graded(ps):
    """Samples that have real (non-null) text ground truth."""
    return {k: v for k, v in ps.items() if v["gt_text"]}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def approach_metrics(ps, getter):
    """getter(sample)->(text, fail:bool, char_acc:float, exact:bool)."""
    graded_ps = graded(ps)
    n = len(graded_ps)
    if n == 0:
        return None
    texts, fails, cas, exacts = [], [], [], []
    for v in graded_ps.values():
        text, fail, ca, ex = getter(v)
        texts.append(text)
        fails.append(fail)
        cas.append(ca)
        exacts.append(ex)
    return {
        "n": n,
        "exact_rate": sum(exacts) / n,
        "char_acc": mean(cas),
        "fail_rate": sum(fails) / n,
        "wrong_rate": 1 - sum(exacts) / n,
    }


def main():
    r = load()
    ps = r["per_sample"]
    vpool = r["val_pool_behavioral"]
    g = graded(ps)

    # ── approach table ──────────────────────────────────────────────────
    def baseline_getter(v):
        b = v["baseline"]
        return b["final_text"], (b["final_text"] == ""), b["char_acc"], b["exact"]

    def best_variant_getter(v):
        best = max(v["variants"].items(), key=lambda kv: kv[1]["char_acc"])
        return best[1]["text"], best[1]["fail"], best[1]["char_acc"], best[1]["exact"]

    def multipass_getter(v):
        # oracle upper bound: did ANY pass produce GT?
        produced = v["multipass"]["gt_produced"]
        text = "(oracle)" if produced else v["baseline"]["final_text"]
        return text, (not produced), v["multipass"]["gt_sim_best"], produced

    def consensus_getter(v):
        c = v["consensus_and_ranking"]["consensus"]
        return c["text"], (c["text"] == ""), c["char_acc"], c["exact"]

    def rank_charconf_getter(v):
        c = v["consensus_and_ranking"]["rank_charconf"]
        return c["text"], (c["text"] == ""), c["char_acc"], c["exact"]

    def rank_consensus_getter(v):
        c = v["consensus_and_ranking"]["rank_consensus"]
        return c["text"], (c["text"] == ""), c["char_acc"], c["exact"]

    approaches = {
        "Baseline (production)": approach_metrics(ps, baseline_getter),
        "Best preprocessing (per-sample oracle)": approach_metrics(ps, best_variant_getter),
        "Multi-pass (any-pass oracle)": approach_metrics(ps, multipass_getter),
        "Consensus (char-aware vote)": approach_metrics(ps, consensus_getter),
        "Rank: char-confidence": approach_metrics(ps, rank_charconf_getter),
        "Rank: consensus": approach_metrics(ps, rank_consensus_getter),
    }

    # per-preprocessing-variant accuracy (graded samples)
    variant_rows = []
    vnames = list(next(iter(g.values()))["variants"].keys())
    for vn in vnames:
        cas, exs, fails, lats = [], [], [], []
        for v in g.values():
            d = v["variants"][vn]
            cas.append(d["char_acc"])
            exs.append(d["exact"])
            fails.append(d["fail"])
            lats.append(d["lat"])
        variant_rows.append({
            "variant": vn,
            "n": len(g),
            "exact_rate": sum(exs) / len(g),
            "char_acc": mean(cas),
            "fail_rate": mean(fails),
            "avg_latency_ms": round(mean(lats) * 1000, 1),
        })

    # ── per_sample_results.json ─────────────────────────────────────────
    per_out = {}
    for pid, v in ps.items():
        per_out[pid] = {
            "gt_text": v["gt_text"], "readable": v["readable"],
            "layout": v["layout"], "crop": f"{v['crop_w']}x{v['crop_h']}",
            "aspect": v["aspect"],
            "baseline_text": v["baseline"]["final_text"],
            "baseline_conf": v["baseline"]["final_conf"],
            "baseline_exact": v["baseline"]["exact"],
            "baseline_char_acc": v["baseline"]["char_acc"],
            "multipass_gt_produced": v["multipass"]["gt_produced"],
            "multipass_best_sim": v["multipass"]["gt_sim_best"],
            "variants": {vn: d["text"] for vn, d in v["variants"].items()},
            "consensus": v["consensus_and_ranking"]["consensus"]["text"],
            "rank_charconf": v["consensus_and_ranking"]["rank_charconf"]["text"],
        }
    (bc.OUT_DIR / "per_sample_results.json").write_text(
        json.dumps(per_out, indent=2), encoding="utf-8")

    # save a machine-readable metrics blob for the writer step
    metrics = {
        "approaches": approaches,
        "variant_rows": variant_rows,
        "val_pool": vpool,
        "graded_ids": list(g.keys()),
    }
    (bc.OUT_DIR / "_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("metrics + per_sample written")
    print(json.dumps(metrics["approaches"], indent=1))


if __name__ == "__main__":
    main()
