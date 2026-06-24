#!/usr/bin/env python3
"""
Roofline plot for the iterative-SpMV kernels (docs/ROOFLINE.md).

Draws the classic log-log roofline (like the SPM-notes figure):
  * x-axis = arithmetic intensity (FLOP/byte),
  * y-axis = attainable performance (GFLOP/s),
  * a sloped "memory-bandwidth ceiling" (y = peak_BW * AI) that bends at the
    "ridge point" into a flat "compute ceiling" (y = peak compute),
  * each measured kernel drawn as a point: where it sits tells the story
    (left of the ridge = memory regime; touching the diagonal = at the
    bandwidth wall; well below = headroom / overhead-bound).

Data below is the post-fix cluster run (node01, 2026-06-23) already recorded in
docs/ROOFLINE.md. Edit the DATA dict if you re-run the benchmark.

Run (uses the project's existing venv):
    .venv_pdf/bin/python scripts/plot_roofline.py
Output: docs/roofline.png  (and docs/roofline_1thread.png for context)
"""

import os
import matplotlib
matplotlib.use("Agg")               # no display needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "docs")

# --- measured data (post-fix run, node01, 2026-06-23) -------------------------
# ceilings per thread count: (peak_compute_GFLOPs, peak_BW_GBs)
CEIL = {
    1:  (19.6,  9.9),
    32: (287.3, 49.2),   # peak BW = mean of the four 32-thread cells (~48-50)
}

# K1 (fused SpMV + norm), 32 threads: per matrix
#   label -> (achieved_GFLOPs, AI_uncached, AI_cached)
K1_32 = {
    "100k / 4M  (40 nnz/row)":  (5.60, 0.1005, 0.1653),
    "100k / 20M (200 nnz/row)": (6.12, 0.1001, 0.1664),
    "500k / 4M  (8 nnz/row)":   (4.60, 0.1023, 0.1607),
    "500k / 20M (40 nnz/row)":  (5.34, 0.1005, 0.1653),
}
# K2 (scaling pass), 32 threads: achieved GFLOP/s at fixed AI = 0.0625
K2_32 = {
    "100k / 4M":  (0.2966, 0.0625),
    "100k / 20M": (0.2872, 0.0625),
    "500k / 4M":  (0.5826, 0.0625),
    "500k / 20M": (0.3048, 0.0625),
}

# --- cache-aware (hierarchical) roofline data --------------------------------
# Hierarchical roofline at the cluster's 32 threads. ALL values MEASURED on
# node01 (post-fix run + --cache-sweep, 2026-06-23). Per-level bandwidths are the
# plateaus read off the 32-thread sweep (SWEEP below); DRAM uses the STREAM peak.
HIER = {
    "label":        "node01, 32 threads (measured)",
    "peak_compute": 287.3,                                       # GFLOP/s
    "levels":       {"L1": 1000.0, "L2": 580.0, "L3": 325.0, "DRAM": 51.0},
    "estimated":    set(),                                       # all measured -> solid lines
    # representative K1 (dense 100k/20M, the clearest overshoot) and K2, 32 threads:
    "k1":           (6.12, 0.1001, 0.1664),                      # gflops, AI_uncached, AI_cached
    "k2":           (0.2854, 0.0625),
}

# raw bandwidth-vs-working-set sweep (cluster node01, 32 threads): (buf KB, GB/s)
SWEEP = [
    (2, 984.5), (4, 1018.6), (8, 615.0), (16, 607.4), (32, 555.3), (64, 357.9),
    (128, 330.5), (256, 322.4), (512, 172.9), (1024, 59.9), (2048, 59.7),
    (4096, 59.7), (8192, 59.1), (16384, 58.4), (32768, 59.1),
]
CACHE_KB = {"L1": 32, "L2": 256, "L3": 20480}   # sysconf data-cache sizes (node01)


def roofline_xy(peak_compute, peak_bw, x_lo, x_hi):
    """Return the polyline (xs, ys) of the roofline = min(peak_compute, bw*AI)."""
    ridge = peak_compute / peak_bw
    xs = [x_lo, ridge, x_hi]
    ys = [peak_bw * x_lo, peak_compute, peak_compute]
    return xs, ys, ridge


def draw(threads, k1, k2, outfile, also_1t=True):
    peak_c, peak_bw = CEIL[threads]
    x_lo, x_hi = 0.04, 20.0
    y_lo, y_hi = 0.2, 500.0

    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    # shaded memory-bound region (AI left of the ridge)
    ridge = peak_c / peak_bw
    ax.axvspan(x_lo, ridge, color="#f2f2f2", zorder=0)
    ax.text(x_lo * 1.15, y_hi * 0.55, "memory-bound region\n(AI < ridge)",
            fontsize=8, color="#888888", va="top")

    # main roofline (this thread count)
    xs, ys, ridge = roofline_xy(peak_c, peak_bw, x_lo, x_hi)
    ax.plot(xs, ys, "-", lw=2.4, color="#222222",
            label=f"roofline @ {threads} threads")
    ax.plot([ridge], [peak_c], "o", color="#222222", ms=5)
    ax.annotate(f"ridge AI = {ridge:.1f}", (ridge, peak_c),
                textcoords="offset points", xytext=(6, -14), fontsize=8)
    ax.text(x_hi * 0.45, peak_c * 1.07, f"peak compute = {peak_c:.0f} GFLOP/s",
            fontsize=8, ha="center")
    # bandwidth-slope label, placed along the diagonal
    ax.text(0.42, peak_bw * 0.42 * 0.55, f"DRAM BW = {peak_bw:.0f} GB/s",
            fontsize=8, rotation=33, color="#444444")

    # optional faint 1-thread roofline for context
    if also_1t and threads != 1:
        pc1, bw1 = CEIL[1]
        xs1, ys1, _ = roofline_xy(pc1, bw1, x_lo, x_hi)
        ax.plot(xs1, ys1, "--", lw=1.2, color="#bbbbbb",
                label="roofline @ 1 thread")

    # K1 points with the AI uncertainty band (uncached -> cached AI)
    first = True
    for lab, (g, ai_lo, ai_hi) in k1.items():
        ax.errorbar(ai_lo, g, xerr=[[0.0], [ai_hi - ai_lo]],
                    fmt="o", ms=7, color="#c0392b", ecolor="#c0392b",
                    elinewidth=1.4, capsize=3,
                    label="K1: SpMV (AI band: x uncached→cached)" if first else None,
                    zorder=5)
        first = False
    # one shared annotation for the K1 cluster
    ax.annotate("K1 (SpMV)\n~60-90% of BW roof",
                (0.10, 5.4), textcoords="offset points", xytext=(20, 26),
                fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

    # K2 points
    first = True
    for lab, (g, ai) in k2.items():
        ax.plot(ai, g, "s", ms=7, color="#1f7a8c",
                label="K2: scaling pass" if first else None, zorder=5)
        first = False
    ax.annotate("K2 (scaling)\noverhead/latency-bound,\nfar below roof",
                (0.0625, 0.35), textcoords="offset points", xytext=(24, -6),
                fontsize=9, color="#1f7a8c",
                arrowprops=dict(arrowstyle="->", color="#1f7a8c", lw=1))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Arithmetic intensity  [FLOP / byte]")
    ax.set_ylabel("Performance  [GFLOP/s]")
    ax.set_title(f"Roofline — iterative-SpMV kernels (node01, {threads} threads)")
    ax.grid(True, which="both", ls=":", lw=0.5, color="#dddddd")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out = os.path.join(OUTDIR, outfile)
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


def _save(fig, outfile):
    out = os.path.join(OUTDIR, outfile)
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {out} (+ .pdf)")


def draw_hierarchical(hier, outfile):
    """Cache-aware roofline: one sloped ceiling per memory level (L1/L2/L3/DRAM)
    plus the flat compute ceiling, with the kernels placed on top."""
    peak_c = hier["peak_compute"]
    levels = hier["levels"]
    estimated = hier.get("estimated", set())
    x_lo, x_hi = 0.04, 20.0
    y_lo, y_hi = 0.1, max(500.0, peak_c * 1.5)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    xs = [x_lo, x_hi]

    # flat compute ceiling
    ax.plot(xs, [peak_c, peak_c], "-", lw=2.2, color="#222222",
            label=f"compute ceiling ({peak_c:.0f} GFLOP/s)")

    # one sloped ceiling per level: y = min(compute, bw * AI)
    colors = {"L1": "#1b9e77", "L2": "#7570b3", "L3": "#e6ab02", "DRAM": "#d95f02"}
    for name in ("L1", "L2", "L3", "DRAM"):
        if name not in levels:
            continue
        bw = levels[name]
        ys = [min(peak_c, bw * x_lo), min(peak_c, bw * x_hi)]
        est = name in estimated
        ax.plot(xs, ys, "--" if est else "-", lw=1.8, color=colors[name],
                label=f"{name} BW {'≈' if est else '='} {bw:.0f} GB/s{' (est.)' if est else ''}")

    # kernels
    g1, ai_lo, ai_hi = hier["k1"]
    ax.errorbar(ai_lo, g1, xerr=[[0.0], [ai_hi - ai_lo]], fmt="o", ms=8,
                color="#c0392b", ecolor="#c0392b", elinewidth=1.6, capsize=3,
                label="K1: SpMV (AI band x uncached→cached)", zorder=6)
    ax.annotate("K1 sits ABOVE the DRAM line\nbut BELOW L3/L2  →  fed from cache (x reuse)",
                (ai_lo, g1), textcoords="offset points", xytext=(18, 18),
                fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    g2, ai2 = hier["k2"]
    ax.plot(ai2, g2, "s", ms=8, color="#1f7a8c", label="K2: scaling pass", zorder=6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Arithmetic intensity  [FLOP / byte]")
    ax.set_ylabel("Performance  [GFLOP/s]")
    ax.set_title(f"Cache-aware (hierarchical) roofline — {hier['label']}")
    ax.grid(True, which="both", ls=":", lw=0.5, color="#dddddd")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    _save(fig, outfile)


def draw_cache_curve(sweep, cache_kb, outfile):
    """Raw bandwidth vs working-set-size curve (the cache hierarchy directly)."""
    xs = [s for s, _ in sweep]
    ys = [g for _, g in sweep]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(xs, ys, "-o", ms=4, color="#222222")
    for name, kb in cache_kb.items():
        ax.axvline(kb, ls="--", lw=1, color="#aaaaaa")
        ax.annotate(f"{name}\n({kb} KB)", (kb, max(ys) * 0.92),
                    fontsize=8, color="#777777", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("per-thread buffer size  [KB]   (3 buffers ⇒ level crossed near size/3)")
    ax.set_ylabel("achieved bandwidth  [GB/s]")
    ax.set_title("Bandwidth vs working-set size (reveals the cache hierarchy)")
    ax.grid(True, which="both", ls=":", lw=0.5, color="#dddddd")
    fig.tight_layout()
    _save(fig, outfile)


if __name__ == "__main__":
    draw(32, K1_32, K2_32, "roofline.png", also_1t=True)   # single-DRAM (cluster 32t)
    draw_hierarchical(HIER, "roofline_hier.png")           # cache-aware roofline
    draw_cache_curve(SWEEP, CACHE_KB, "cache_bandwidth.png")
    print("done")
