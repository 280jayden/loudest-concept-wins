"""
Final figures for the executive summary. Local, no GPU.

Replaces the earlier three-figure set. The first two figures there plotted the same
thing (detection against concept share) with different lines on each, which reads as a
repeated chart. They are merged into one panel carrying all three methods.

  fig_a  detection against concept share, all three methods, with both floors
  fig_b  what the description names at each ratio, as counts rather than stacked shares

Everything is computed from the pickles, so a figure cannot drift from the table it
illustrates. Raw randomly-sampled descriptions are printed at the end for the write-up,
seeded so the selection is reproducible and not chosen by eye.
"""
import pickle, json, os, random
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results/RESULTS"
OUT = "../results/figures"
os.makedirs(OUT, exist_ok=True)
THR = 0.3
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SHARE = {0.0: 100, 0.25: 75, 0.5: 50, 0.75: 25, 0.9: 10, 1.0: 0}

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
})
SA, LR, NLA, GREY = "#A93E2B", "#2C5F73", "#43744A", "#8A8A85"


def llama():
    d = pickle.load(open(f"{R}/trained_magnitude.pkl", "rb"))
    out = {}
    for tag in ("sa", "lr"):
        for al in ALPHAS:
            rows = [r for k, v in d.items() if k[0] == tag and k[2] == al for r in v]
            if not rows:
                continue
            c = Counter()
            for r in rows:
                a, b = r["hit_A"] >= THR, r["hit_B"] >= THR
                c["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
            out[(tag, al)] = (c, len(rows))
    return out


def nla():
    j = json.load(open(f"{R}/nla_full_curve_v2.json"))
    key = {100: "100%", 75: "75%", 50: "50%", 25: "25%", 10: "9%"}
    return {s: j[k] for s, k in key.items() if k in j}


# --------------------------------------------------------------- figure A
def fig_a(L, N):
    """All three methods on one axis, each with its own false-positive floor."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    shares = [100, 75, 50, 25, 10]
    xs = list(range(len(shares)))
    als = [0.0, 0.25, 0.5, 0.75, 0.9]

    series = []
    for tag, col, lab, mk in [("sa", SA, "Llama, scalar affine (4,097 params)", "o"),
                              ("lr", LR, "Llama, + rank-16 (135,169 params)", "^")]:
        ys, ann = [], []
        for al in als:
            c, n = L[(tag, al)]
            b = c["B"] + c["both"]
            ys.append(100 * b / n); ann.append(f"{b}/{n}")
        series.append((ys, ann, col, lab, mk))
        ax.plot(xs, ys, mk + "-", color=col, lw=2, ms=5, label=lab, zorder=3)

    ys, ann = [], []
    for s in shares:
        r = N[s]
        ys.append(100 * r["B"] / r["n"]); ann.append(f"{r['B']}/{r['n']}")
    ax.plot(xs, ys, "s-", color=NLA, lw=2, ms=5,
            label="Gemma, NLA verbaliser", zorder=3)
    nla_floor = 100 * N[100]["A"] / N[100]["n"]

    # floors
    ax.axhline(nla_floor, color=NLA, ls=":", lw=1.2, zorder=1)
    ax.text(0.04, nla_floor + 2.4, f"NLA reports an absent concept {nla_floor:.0f}% of the time",
            fontsize=6.8, color=NLA)
    ax.axhline(0, color=SA, ls=":", lw=1.2, zorder=1)
    ax.text(0.04, -9.5, "Llama adapters report an absent concept 0% of the time",
            fontsize=6.8, color=SA)

    # annotate only where it matters: the collapse
    for i, s in enumerate(shares):
        if s not in (50, 25):
            continue
        for ys_, ann_, col, _, _ in series:
            off = (-15, -14) if col == SA else (26, 4)
            ax.annotate(ann_[i], (i, ys_[i]), textcoords="offset points",
                        xytext=off, ha="center", fontsize=6.8, color=col)
    ax.annotate(f"{N[25]['B']}/{N[25]['n']}", (3, 100 * N[25]["B"] / N[25]["n"]),
                textcoords="offset points", xytext=(28, 2), ha="center",
                fontsize=6.8, color=NLA)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}%" for s in shares])
    ax.set_xlabel("share of the activation held by the second concept")
    ax.set_ylabel("% of descriptions naming it")
    ax.set_ylim(-13, 108)
    ax.legend(fontsize=7.2, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    ax.set_title("A better method moves the threshold, and does not remove it",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_a_all_interpreters.png")
    plt.close(fig)


# --------------------------------------------------------------- figure B
def fig_b(L, N):
    """What each description named: concept A only, concept B only, or both.

    Categories are exclusive, so with "neither" they sum to the sample. Percentages
    rather than counts, because the two samples differ (240 against 200). "Neither" is
    reported in the caption instead of a fourth bar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
    show = [0.25, 0.5, 0.75, 0.9]
    shares = [75, 50, 25, 10]
    xs = list(range(len(show)))
    w = 0.27

    A_ONLY, B_ONLY, BOTH = "#2C5F73", "#A93E2B", "#8C4A9E"

    llama = [(100 * L[("sa", a)][0]["A"] / L[("sa", a)][1],
              100 * L[("sa", a)][0]["B"] / L[("sa", a)][1],
              100 * L[("sa", a)][0]["both"] / L[("sa", a)][1]) for a in show]
    nlarows = []
    for s_ in shares:
        r = N[s_]
        nlarows.append((100 * (r["A"] - r["both"]) / r["n"],
                        100 * (r["B"] - r["both"]) / r["n"],
                        100 * r["both"] / r["n"]))

    for ax, rows, title, tot in [(axes[0], llama, "Llama, scalar affine adapter", 240),
                                 (axes[1], nlarows, "Gemma, NLA verbaliser", 200)]:
        a_ = [r[0] for r in rows]; b_ = [r[1] for r in rows]; bo = [r[2] for r in rows]
        ax.bar([x - w for x in xs], a_, w, color=A_ONLY, label="concept A only")
        ax.bar(xs, b_, w, color=B_ONLY, label="concept B only")
        ax.bar([x + w for x in xs], bo, w, color=BOTH, label="both")
        for i in range(len(xs)):
            for off, v, col in [(-w, a_[i], A_ONLY), (0, b_[i], B_ONLY), (w, bo[i], BOTH)]:
                ax.text(i + off, v + 2.5, f"{v:.0f}", ha="center", fontsize=7,
                        color=col, fontweight="bold" if col == BOTH else "normal")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s_}%" for s_ in shares])
        ax.set_title(f"{title}  (n={tot})", fontsize=9, loc="left")
        ax.set_ylim(0, 108)
        ax.grid(axis="x", alpha=0)

    axes[0].set_ylabel("% of descriptions")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.8, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.supxlabel("share of the activation held by the second concept (concept B)",
                  fontsize=9, y=0.10)
    fig.suptitle("The adapter almost never names two concepts; the verbaliser often does",
                 fontsize=9.5, x=0.011, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    fig.savefig(f"{OUT}/fig_b_one_concept.png")
    plt.close(fig)


# --------------------------------------------------------------- raw examples
def raw_examples(seed=0, k=4):
    d = pickle.load(open(f"{R}/trained_magnitude.pkl", "rb"))
    random.seed(seed)
    print("\n" + "=" * 74)
    print(f"RANDOM DESCRIPTIONS  (random.seed({seed}), not selected by eye)")
    print("=" * 74)
    for al in [0.5, 0.75, 0.9]:
        pool = [(k_[1], r) for k_, v in d.items() if k_[0] == "sa" and k_[2] == al
                for r in v]
        print(f"\n--- second concept at {SHARE[al]}% share "
              f"({k} random of {len(pool)}) ---")
        for nm, r in random.sample(pool, k):
            a = "Y" if r["hit_A"] >= THR else "n"
            b = "Y" if r["hit_B"] >= THR else "n"
            print(f"  [A={a} B={b}]  {nm[:44]}")
            print(f"      {r['label'][:96]!r}")


if __name__ == "__main__":
    L, N = llama(), nla()
    fig_a(L, N)
    fig_b(L, N)
    print("wrote:", sorted(f for f in os.listdir(OUT) if f.startswith("fig_")))
    raw_examples()
