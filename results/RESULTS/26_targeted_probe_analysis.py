"""
Targeted probing: does NAMING the concept in the question recover it?

This is the experiment that answers "isn't this obvious?". Open-ended description is
a discovery task, and one could argue a quiet concept is simply hard to notice. But a
real monitor KNOWS what it is looking for. So we ask the model directly:
    "Does <activation> contain anything related to <concept B>?"

TWO TRAPS, BOTH HANDLED

1. hit_B IS CONTAMINATED FOR THIS PROBE.
   The question names concept B, so the model's answer contains the concept's words
   whether or not it is present. Scoring that text behaviourally then "detects" B on
   pure-A activations 52/90 = 58% of the time. hit_B is therefore meaningless for the
   targeted and forced probes; only the yes/no answer counts. (For the OPEN probe the
   question never names B, so hit_B stays valid there.)

2. A "yes" IS NOT EVIDENCE ON ITS OWN.
   The model may agree with any leading question, so we ask the same question about B
   while feeding a PURE A vector, where B is genuinely absent. That yes-rate is the
   floor every other number is measured against.

WHICH CONTROL FILE IS VALID
targeted_probe.pkl also holds CTRL rows, but their anchor-described rate is 0/90 - the
injection was dead in that run (a pickle load-modify-dump race). The separate
targeted_probe_CTRL.pkl run shows 73/90 = 81%, in line with the live cells (83-92%).
The separate file is the valid control; the in-file CTRL rows are discarded.
"""
import pickle, json, math
from collections import defaultdict

MAIN = "targeted_probe.pkl"
CTRL = "targeted_probe_CTRL.pkl"
THR = 0.3


def two_prop_z(h1, n1, h2, n2):
    """z and two-sided p for h1/n1 vs h2/n2."""
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pv


def load(dirpath="."):
    d = pickle.load(open(f"{dirpath}/{MAIN}", "rb"))
    c = pickle.load(open(f"{dirpath}/{CTRL}", "rb"))
    d = {k: v for k, v in d.items() if str(k[1]) != "CTRL_pureA"}   # drop dead-injection rows
    return d, c


def summarise(d, c):
    yes = defaultdict(lambda: [0, 0])       # (alpha, probe) -> yes, n
    beh = defaultdict(lambda: [0, 0])       # (alpha, probe) -> hit_B, n
    anc = defaultdict(lambda: [0, 0])
    for (nm, al, pr, sc), v in d.items():
        for r in v["rows"]:
            k = (str(al), pr)
            beh[k][1] += 1; beh[k][0] += (r["hit_B"] >= THR)
            anc[k][1] += 1; anc[k][0] += (r["hit_A"] >= THR)
            if r["yes"] is not None:
                yes[k][1] += 1; yes[k][0] += bool(r["yes"])
    cy = [0, 0]; cb = [0, 0]
    for v in c.values():
        for r in v["rows"]:
            cb[1] += 1; cb[0] += (r["hit_B"] >= THR)
            if r["yes"] is not None:
                cy[1] += 1; cy[0] += bool(r["yes"])
    return yes, beh, anc, cy, cb


if __name__ == "__main__":
    d, c = load()
    yes, beh, anc, cy, cb = summarise(d, c)
    ALS = ["0.5", "0.75", "0.9"]
    SHARE = {"0.5": "50%", "0.75": "25%", "0.9": "10%"}

    print("CONTROL, pure A (concept B genuinely absent):")
    print(f"   says yes            {cy[0]}/{cy[1]} = {cy[0]/cy[1]*100:.0f}%   <- the floor")
    print(f"   hit_B fires anyway  {cb[0]}/{cb[1]} = {cb[0]/cb[1]*100:.0f}%   <- why hit_B is unusable here")

    print("\n=== OPEN probe (question never names B, so hit_B is valid) ===")
    print(f"{'share of B':>12}{'B named':>14}")
    for al in ALS:
        h, n = beh[(al, "open")]
        print(f"{SHARE[al]:>12}{h:>7}/{n:<4} {h/n*100:>3.0f}%")

    print("\n=== TARGETED probe (B is named in the question; only yes/no counts) ===")
    print(f"{'share of B':>12}{'says yes':>14}{'vs floor':>11}{'z':>7}{'p':>9}")
    for al in ALS:
        h, n = yes[(al, "targeted")]
        z, pv = two_prop_z(h, n, cy[0], cy[1])
        print(f"{SHARE[al]:>12}{h:>7}/{n:<4} {h/n*100:>3.0f}%"
              f"{h/n*100-cy[0]/cy[1]*100:>+10.0f}{z:>7.2f}{pv:>9.3f}")

    print("\n=== FORCED probe ('this contains two concepts, one may relate to B') ===")
    print("   hit_B is contaminated the same way; reported for completeness only")
    for al in ALS:
        h, n = beh[(al, "forced")]
        print(f"{SHARE[al]:>12}{h:>7}/{n:<4} {h/n*100:>3.0f}%")

    print("\n=== sanity: was the anchor described? (injection working) ===")
    for al in ALS:
        for pr in ["open", "targeted", "forced"]:
            h, n = anc[(al, pr)]
            print(f"   {SHARE[al]:>4} {pr:9} {h}/{n} = {h/n*100:.0f}%")

    out = {"control_yes": cy, "control_hitB_contamination": cb,
           "open_behavioural": {SHARE[a]: beh[(a, "open")] for a in ALS},
           "targeted_yes": {SHARE[a]: yes[(a, "targeted")] for a in ALS}}
    json.dump(out, open("targeted_probe_clean.json", "w"), indent=1)
