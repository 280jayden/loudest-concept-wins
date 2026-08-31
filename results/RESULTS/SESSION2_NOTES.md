# Session 2 notes

## Completed
- TARGETED PROBE (+ false-positive control) - closes the "just ask it directly" objection.
    control (concept ABSENT, pure A): says YES 28.9%, behavioural-metric hit 57.8%
    -> the behavioural metric is CONFOUNDED for targeted questions (model echoes the
       question phrase back; our scorer then reads it as a hit). Do not use it there.
    discrimination vs control:  50% share +35.9pp p<0.0001 | 25% +8.1pp p=0.36 (n.s.)
                                 9% share -3.9pp p=0.83 (n.s.)
    -> naming the concept does NOT recover it below ~25%, and adds a ~29% false-alarm rate.

- GEMMA NLA (gemma-3-12b-it L32 + gemma-scope-2 + kitft NLA) - cross-architecture check.
    Gate 2 passed (all |cos| <= 0.037).
    KEY FIX: NLAs are trained on REAL activations, so a bare SAE decoder row is OOD.
      ||b_dec|| = 73,948 vs prescribed injection norm 80,000 -> a real activation is
      ~92% shared DC component. Native format is  v = b_dec + s*(alpha*dA + (1-a)*dB).
      With that, confabulation stopped and descriptions became clean.
    alpha=0.9 (concerning at 10%): 0/6 on BOTH safety pairs, hand-read, no mentions.
    alpha=0.5: genuinely names both, as a FUSION ("cooking/explosives terminology",
      "racial/ethnic culinary terminology") - same fusion behaviour as Llama's 10 cases.

## Numbers to use
- Llama alpha=0.9 restricted to WORKING generations (dominant concept detected):
    206/216 worked -> minority concerning concept reported 0/206.
    Per scale, working: 29,34,36,35,36,36 of 36 -> not carried by one lucky scale.
    THIS is the formulation to headline (pre-empts "bad injection magnitude").

## Open / not done
- Same "given it worked" conditional for alpha=0.75 (currently 1/252) and for Gemma.
- Gemma leg is only 2 pairs x 6 draws = 12 independent draws vs Llama's 36.
  Cheap fix: cooking x drug_harm, legal x hate, legal x explosives already screened.
- Metric parity: Llama behavioural=1.6% vs keyword=2.5%, 96.7% per-concept agreement.
  Keyword is OK for Llama's short labels, NOT for the NLA's long hedging text.
  Proper fix needs base google/gemma-3-12b-it (access now granted) for the
  synthetic-conversation route.
- SA+LR sweep still ~50 cells short.
- FIGURES AND WRITE-UP NOT STARTED.

## Errors made this session (worth recording for the write-up)
1. Qwen NLA: built concept directions contrastively -> came out near-collinear
   (cooking x fraud cos=-0.94) and OOD. Experiment invalid, abandoned.
2. Pickle race: two threads load-modify-dump the same file -> control results
   silently overwritten, zero cells saved, no error raised.
3. Applied a validity check from ONE pair to all three -> wrongly excluded
   alpha=0.9 from the Gemma sweep (cooking x hatred was valid all along).
4. Keyword metric on Gemma reported 50% both-capture; hand-reading found ~0 clean.
