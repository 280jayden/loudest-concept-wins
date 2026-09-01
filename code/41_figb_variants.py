"""Figure B in three forms so the clearest can be picked: one panel per interpreter,
and the same data as a table."""
import pickle, json, os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT, THR = "../results/RESULTS", "../results/figures", 0.3
SHOW = [0.25, 0.5, 0.75, 0.9]
SHARES = [75, 50, 25, 10]
A_ONLY, B_ONLY, BOTH = "#2C5F73", "#A93E2B", "#8C4A9E"
plt.rcParams.update({"figure.dpi": 200, "font.size": 9, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6})


def llama():
    d = pickle.load(open(f"{R}/trained_magnitude.pkl", "rb"))
    out = []
    for al in SHOW:
        rows = [r for k, v in d.items() if k[0] == "sa" and k[2] == al for r in v]
        c = Counter()
        for r in rows:
            a, b = r["hit_A"] >= THR, r["hit_B"] >= THR
            c["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
        out.append((c["A"], c["B"], c["both"], c["neither"], len(rows)))
    return out


def nla():
    j = json.load(open(f"{R}/nla_full_curve_v2.json"))
    key = {75: "75%", 50: "50%", 25: "25%", 10: "9%"}
    out = []
    for s in SHARES:
        r = j[key[s]]
        out.append((r["A"] - r["both"], r["B"] - r["both"], r["both"], r["neither"], r["n"]))
    return out


def panel(rows, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs, w = list(range(len(SHARES))), 0.27
    a_ = [100 * r[0] / r[4] for r in rows]
    b_ = [100 * r[1] / r[4] for r in rows]
    bo = [100 * r[2] / r[4] for r in rows]
    ax.bar([x - w for x in xs], a_, w, color=A_ONLY, label="concept A only")
    ax.bar(xs, b_, w, color=B_ONLY, label="concept B only")
    ax.bar([x + w for x in xs], bo, w, color=BOTH, label="both")
    for i in range(len(xs)):
        for off, v, col in [(-w, a_[i], A_ONLY), (0, b_[i], B_ONLY), (w, bo[i], BOTH)]:
            ax.text(i + off, v + 2.5, f"{v:.0f}%", ha="center", fontsize=7, color=col,
                    fontweight="bold" if col == BOTH else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}%" for s in SHARES])
    ax.set_xlabel("share held by the second concept (B)")
    ax.set_ylabel("% of descriptions")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", alpha=0)
    ax.legend(fontsize=7.4, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.20))
    ax.set_title(title, fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}")
    plt.close(fig)


if __name__ == "__main__":
    L, N = llama(), nla()
    panel(L, "Llama, scalar affine adapter (n=240)", "fig_b1_llama.png")
    panel(N, "Gemma, NLA verbaliser (n=200)", "fig_b2_nla.png")
    print("wrote fig_b1_llama.png, fig_b2_nla.png\n")

    for name, rows in [("Llama, scalar affine adapter (n=240)", L),
                       ("Gemma, NLA verbaliser (n=200)", N)]:
        print(name)
        print(f"{'B share':>9}{'A only':>14}{'B only':>14}{'both':>13}{'neither':>13}")
        for s, (a, b, bo, ne, n) in zip(SHARES, rows):
            f = lambda v: f"{v}/{n} ({100*v/n:.0f}%)"
            print(f"{str(s)+'%':>9}{f(a):>14}{f(b):>14}{f(bo):>13}{f(ne):>13}")
        print()
