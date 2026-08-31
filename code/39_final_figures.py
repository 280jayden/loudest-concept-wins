"""
Figures for the write-up, built from the final data. Runs locally, no GPU.

Three figures, one per key experiment, as the application format asks for:
  fig1  the main curve, both Llama adapters, with the concept-absent control
  fig2  what the description names (A only / B only / both / neither) across ratios
  fig3  cross-architecture, Llama against the NLA, each with its own floor

Everything is computed from the pickles rather than typed in, so a figure cannot drift
from the table it illustrates.
"""
import pickle, json, os
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
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
SA, LR, NLA = "#B4432F", "#2C5F73", "#4B7F52"
GREY = "#8A8A85"


def llama_counts():
    """{(adapter, alpha): Counter of A/B/both/neither} plus totals."""
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


def nla_counts():
    j = json.load(open(f"{R}/nla_full_curve_v2.json"))
    key = {100: "100%", 75: "75%", 50: "50%", 25: "25%", 10: "9%"}
    return {s: j[k] for s, k in key.items() if k in j}


# ----------------------------------------------------------------- figure 1
def fig1(L):
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    xs = [SHARE[a] for a in ALPHAS]
    for tag, col, lab in [("sa", SA, "scalar affine, 4,097 params"),
                          ("lr", LR, "+ rank-16, 135,169 params")]:
        ys, ann = [], []
        for al in ALPHAS:
            c, n = L[(tag, al)]
            b = c["B"] + c["both"]
            ys.append(100 * b / n)
            ann.append(f"{b}/{n}")
        ax.plot(xs, ys, "o-", color=col, lw=2, ms=5, label=lab, zorder=3)
        dy = 9 if tag == "lr" else -14
        for x, y, t in zip(xs, ys, ann):
            if x in (100, 50, 25):
                ax.annotate(t, (x, y), textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=7, color=col)

    # the control: same measurement with the concept absent
    c, n = L[("sa", 1.0)]
    ax.scatter([0], [100 * (c["B"] + c["both"]) / n], marker="s", s=42,
               color=GREY, zorder=4, label="concept absent (control)")

    ax.axvspan(0, 27, color=GREY, alpha=0.08, zorder=0)
    ax.text(13.5, 88, "detection at the\ncontrol rate", ha="center", fontsize=7.5,
            color=GREY)
    ax.set_xticks(xs)
    ax.set_xlabel("share of the activation held by the second concept (%)")
    ax.set_ylabel("% of descriptions naming it")
    ax.set_ylim(-10, 104)
    ax.invert_xaxis()
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    ax.set_title("Detection collapses to the control rate below a 25% share",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_main_curve.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 2
def fig2(L):
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    show = [0.0, 0.25, 0.5, 0.75, 0.9]
    xs = list(range(len(show)))
    keys = ["A", "B", "both", "neither"]
    cols = {"A": "#7FA8B8", "B": "#D98A73", "both": "#3F6B45", "neither": "#CFCFC8"}
    labs = {"A": "anchor only", "B": "second concept only",
            "both": "both", "neither": "neither"}
    bottom = [0] * len(show)
    for k in keys:
        vals = []
        for al in show:
            c, n = L[("sa", al)]
            vals.append(100 * c[k] / n)
        ax.bar(xs, vals, bottom=bottom, color=cols[k], label=labs[k],
               width=0.68, edgecolor="white", linewidth=0.8)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if k == "both" and v > 0:
                ax.text(i, b + v / 2, f"{c if False else ''}", ha="center")
        bottom = [b + v for b, v in zip(bottom, vals)]

    for i, al in enumerate(show):
        c, n = L[("sa", al)]
        ax.text(i, 101.5, f"both\n{c['both']}/{n}", ha="center", fontsize=7,
                color="#3F6B45")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{SHARE[a]}%" for a in show])
    ax.set_xlabel("share of the activation held by the second concept")
    ax.set_ylabel("% of descriptions")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", alpha=0)
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.34))
    ax.set_title("Almost every description names exactly one of the two concepts",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_what_it_names.png")
    plt.close(fig)


# ----------------------------------------------------------------- figure 3
def fig3(L, N):
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    shares = [100, 75, 50, 25, 10]
    xs = list(range(len(shares)))

    ys, ann = [], []
    for al in [0.0, 0.25, 0.5, 0.75, 0.9]:
        c, n = L[("sa", al)]
        b = c["B"] + c["both"]
        ys.append(100 * b / n); ann.append(f"{b}/{n}")
    ax.plot(xs, ys, "o-", color=SA, lw=2, ms=5, label="Llama, trained adapter", zorder=3)

    ys2, ann2 = [], []
    for s in shares:
        r = N[s]
        ys2.append(100 * r["B"] / r["n"]); ann2.append(f"{r['B']}/{r['n']}")
    ax.plot(xs, ys2, "s-", color=NLA, lw=2, ms=5,
            label="Gemma, natural language autoencoder", zorder=3)

    ax.axhline(0.0, color=SA, ls=":", lw=1.2)
    ax.text(-0.05, 3.0, "Llama floor 0%", fontsize=7, color=SA, ha="left")
    floor = 100 * N[100]["A"] / N[100]["n"]
    ax.axhline(floor, color=NLA, ls=":", lw=1.2)
    ax.text(-0.05, floor + 3.0, f"NLA floor {floor:.0f}%", fontsize=7,
            color=NLA, ha="left")

    for i, (y, t) in enumerate(zip(ys, ann)):
        if shares[i] in (50, 25, 10):
            ax.annotate(t, (i, y), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color=SA)
    for i, (y, t) in enumerate(zip(ys2, ann2)):
        if shares[i] in (50, 25, 10):
            ax.annotate(t, (i, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=NLA)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}%" for s in shares])
    ax.set_xlabel("share of the activation held by the second concept")
    ax.set_ylabel("% of descriptions naming it")
    ax.set_ylim(-14, 108)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.62))
    ax.set_title("A better interpreter moves the threshold down one step",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_cross_architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    L, N = llama_counts(), nla_counts()
    fig1(L); fig2(L); fig3(L, N)
    print("wrote:", sorted(os.listdir(OUT)))
    print("\nsanity check against the tables:")
    for al in [0.5, 0.75, 0.9, 1.0]:
        c, n = L[("sa", al)]
        print(f"  llama sa {SHARE[al]:>3}%  B={c['B']+c['both']}/{n}  both={c['both']}")
    for s in [50, 25, 10]:
        r = N[s]
        print(f"  nla      {s:>3}%  B={r['B']}/{r['n']}  both={r['both']}")
