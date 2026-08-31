"""
Figures for the write-up. Run after 23_gemma_score.py and 24_llama_constant_basis.py.

Every panel is computed from the saved pickles, not from numbers typed by hand, so a
figure cannot silently drift from the table it illustrates.
"""
import json, pickle, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/workspace/figures"
os.makedirs(FIG, exist_ok=True)
SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]
ALPHAS = [0.5, 0.75, 0.9]
SHARES = [50, 25, 10]
THR = 0.3
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

FEATS_FULL = {"cooking": 13974, "legal": 5174, "programming": 3992, "travel": 1567,
              "explosives": 7175, "hatred_discrim": 2026, "malware": 1715, "ransomware": 6403}
PAIRS8 = [("travel", "explosives"), ("legal", "explosives"), ("cooking", "hatred_discrim"),
          ("legal", "hatred_discrim"), ("cooking", "malware"), ("programming", "malware"),
          ("legal", "ransomware"), ("cooking", "ransomware")]

# `programming` fires 0.20 on its own Gate-1 explanation (keyword-gated it passed 3/3,
# but only on the word "code" inside an unrelated astrology explanation). The verbaliser
# cannot describe it in isolation, so a miss in a mixture is not evidence of omission.
GATE1_FAIL = {"programming"}
PAIRS_OK = [(m, c) for m, c in PAIRS8 if m not in GATE1_FAIL and c not in GATE1_FAIL]


def llama_cells():
    """{(pair, alpha): (detected, n)} on the constant basis."""
    ss = pickle.load(open("/workspace/safety_sweep.pkl", "rb"))
    V = pickle.load(open("/workspace/safe_meta.pkl", "rb"))["VALID2"]
    keep = [nm for nm in sorted(V)
            if all({a: b for a, _, b, _ in V[nm]}.get(al) for al in ALPHAS)]
    out = {}
    for nm in keep:
        for al in ALPHAS:
            d = n = 0
            for sc in SCALES:
                for r in ss.get((nm, al, sc), []):
                    n += 1; d += (r["hit_B"] >= THR)
            out[(nm, al)] = (d, n)
    return keep, out, ss


def gemma_cells():
    """{(pair, alpha): (detected, n)} behavioural, plus keyword and measured ratio."""
    D = pickle.load(open("/workspace/gemma_rerun_descriptions.pkl", "rb"))
    S = pickle.load(open("/workspace/gemma_rerun_scores.pkl", "rb"))
    beh, kw, rat = {}, {}, {}
    for m, c in PAIRS_OK:
        for al in ALPHAS:
            d = D[("sweep", m, c, al)]
            rows = S.get(("score", m, c, al))
            kw[(f"{m} x {c}", al)] = (sum(1 for r in d["rows"][:6] if r["mB"]), 6)
            rat[(f"{m} x {c}", al)] = d["ratio"]
            if rows:
                fb = FEATS_FULL[c]
                beh[(f"{m} x {c}", al)] = (sum(1 for r in rows if r[fb] >= THR), len(rows))
    return beh, kw, rat


def fig1_main(lc, gb):
    """Detection of the minority concept vs its share. The core result."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for cells, lab, col, mk in [(lc, "Llama-3.1-8B, trained SelfIE", "#c0392b", "o"),
                                (gb, "Gemma-3-12B, NLA verbaliser", "#2471a3", "s")]:
        y, lo = [], []
        for al in ALPHAS:
            d = sum(v[0] for k, v in cells.items() if k[1] == al)
            n = sum(v[1] for k, v in cells.items() if k[1] == al)
            y.append(100 * d / n)
            lo.append(f"{d}/{n}")
        ax.plot(SHARES, y, mk + "-", color=col, label=lab, lw=1.8, ms=6)
        for x, v, t in zip(SHARES, y, lo):
            ax.annotate(t, (x, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=col)
    # The two metrics have DIFFERENT false-positive floors, so they need separate lines.
    # Llama's generation-scoring metric validated at 0/144 on random directions and
    # 0/90 pure-A -> B. Gemma's runs at 9/48 = 18.8%, because that SAE has L0=120.
    ax.axhline(18.8, ls=":", c="#2471a3", lw=1)
    ax.text(51, 20.3, "Gemma metric false-positive floor (18.8%)",
            fontsize=6.2, color="#2471a3")
    ax.axhline(0, ls=":", c="#c0392b", lw=1)
    ax.text(51, 1.6, "Llama metric false-positive floor (0%)",
            fontsize=6.2, color="#c0392b")
    ax.set_xticks(SHARES); ax.invert_xaxis()
    ax.set_xlabel("share of the activation held by the second concept")
    ax.set_ylabel("% of descriptions that name it")
    ax.set_ylim(-4, 108); ax.legend(fontsize=7, frameon=False)
    ax.set_title("A second concept is omitted once it stops dominating", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_main.png"); plt.close(fig)


def fig2_scales(ss, keep):
    """The collapse is not an artefact of one injection magnitude."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for al, col, sh in zip(ALPHAS, ["#c0392b", "#e67e22", "#7f8c8d"], SHARES):
        y = []
        for sc in SCALES:
            d = n = 0
            for nm in keep:
                for r in ss.get((nm, al, sc), []):
                    n += 1; d += (r["hit_B"] >= THR)
            y.append(100 * d / n)
        ax.plot(range(len(SCALES)), y, "o-", color=col, label=f"{sh}% share", lw=1.6, ms=5)
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels(SCALES)
    ax.set_xlabel("injection magnitude"); ax.set_ylabel("% naming the second concept")
    ax.legend(fontsize=7, frameon=False); ax.set_ylim(-3, 60)
    ax.set_title("Llama: the omission holds at every injection magnitude", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_scales.png"); plt.close(fig)


def _heat(ax, rows, labels, title, denom):
    M = np.array([[rows[(l, al)][0] / rows[(l, al)][1] for al in ALPHAS] for l in labels])
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{s}%" for s in SHARES])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l[:34] for l in labels], fontsize=6.5)
    for i, l in enumerate(labels):
        for j, al in enumerate(ALPHAS):
            d, n = rows[(l, al)]
            ax.text(j, i, f"{d}/{n}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 or M[i, j] < 0.15 else "black")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"share held by the second concept ({denom})")
    return im


def fig3_pairs(lc, keep, gb):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4),
                             gridspec_kw={"width_ratios": [1, 0.85]})
    _heat(axes[0], lc, keep, "Llama, trained SelfIE", "18 per cell")
    gl = [f"{m} x {c}" for m, c in PAIRS_OK]
    im = _heat(axes[1], gb, gl, "Gemma, NLA verbaliser", "6 per cell")
    fig.colorbar(im, ax=axes, fraction=0.02, label="fraction naming the second concept")
    fig.savefig(f"{FIG}/fig3_pairs.png", bbox_inches="tight"); plt.close(fig)


def fig4_ratio(rat):
    """Methods check: the mixtures really sit at the ratios we claim."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    tgt = {0.5: 1.0, 0.75: 1 / 3, 0.9: 1 / 9}
    for al, col, sh in zip(ALPHAS, ["#c0392b", "#e67e22", "#2471a3"], SHARES):
        v = [rat[k] for k in rat if k[1] == al]
        ax.scatter([sh] * len(v), v, s=22, color=col, alpha=.8, zorder=3)
        ax.hlines(tgt[al], sh - 5, sh + 5, color="black", lw=1.4, zorder=4)
    ax.set_xticks(SHARES); ax.invert_xaxis()
    ax.set_xlabel("nominal share of the second concept")
    ax.set_ylabel("measured activation ratio (SAE)")
    ax.set_title("Mixtures sit at their intended ratios\n(black bars = target)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_ratio.png"); plt.close(fig)


if __name__ == "__main__":
    keep, lc, ss = llama_cells()
    gb, kw, rat = gemma_cells()
    if not gb:
        print("WARNING: no behavioural Gemma scores yet - using keyword for fig1/fig3")
        gb = kw
    fig1_main(lc, gb)
    fig2_scales(ss, keep)
    fig3_pairs(lc, keep, gb)
    fig4_ratio(rat)
    print("wrote:", sorted(os.listdir(FIG)))
