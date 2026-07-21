# Track B / Phase 2 — SOTA Retrospective

*What the ceiling actually was, why the pipeline couldn't beat it, and what the
state-of-the-art move would have been.* Written 2026-07-21, after the deadline,
from the shipped artifacts only (no fresh inference — credits exhausted
2026-05-18; no ground truth — Phase 2 `test.json` answers are `"To be
determined"` placeholders). Every number below is reproducible from files in
this repo against the two measured leaderboard anchors.

---

## TL;DR

The whole effort was capability-bound, not strategy-bound. On the confirmed
metric `LB = mean(Track A IoU, Track B Accuracy)`:

| Submission | LB | TA IoU | TB Acc |
|---|---|---|---|
| random-guessing baseline (matched to `_07`'s cardinality) | ~0.038 | **~0.045** | ~0.03 |
| `_07` | 0.07385 | 0.0677 | 0.08 |
| `v10` (best measured) | 0.09002 | 0.0700 | 0.11 |

The base model was **~1.5× random** on Track A (0.070 vs 0.045) and landed
**11 of 100 correct** on Track B. The entire superstructure — XGBoost LTR,
graph features, anomaly mining, 7-cluster fault heuristic, 5-source ensembles —
moved the score from ~random to ~1.5× random. That is a real but small lift,
and it was near the model's content ceiling on this task.

**The SOTA move was never a better ensemble.** It was (1) cardinality
discipline on Track A, which the pipeline got structurally wrong on 306/500
questions, and (2) real per-question retrieval signal from the production
tool server on Track B, which was never run. Everything else was rearranging
near-random guesses under a metric that punishes hedging.

---

## 1. The metric decides the whole game

`LB = mean(Track A IoU, Track B Accuracy)`, verified exactly against v10
(`(0.0700 + 0.11)/2 = 0.09002`). Two properties dominate every decision:

- **Track A is IoU with small GT.** GT cardinality looks like 1–2 options
  (298/500 questions are tagged `single-answer`, i.e. GT size 1). A prediction
  that hedges wide is capped hard: a 4-option pick containing the one correct
  answer against a size-1 GT scores IoU = 1/4 = **0.25**, versus **1.0** for the
  right single pick. Breadth is actively taxed.
- **Track B is strict Accuracy, not F1.** Extra "recall" lines don't earn
  partial credit; they only add ways to be wrong. Precision-first, single-line
  is the dominant strategy.

Both properties point the same way: **narrow, confident, correctly-sized
answers win; hedging loses.** The pipeline's instinct — ensemble unions, multi-
line fault emissions, 4-option fallbacks — fought the metric.

## 2. Track A was ~1.5× random, and the pipeline knew nothing it could act on

Monte-Carlo random baseline, matched to `_07`'s exact per-question cardinality
(24 options each, GT size 1 for `single-answer`, size 2 for `multiple-answer`):

```
RANDOM baseline TA IoU  ≈ 0.044
_07 MEASURED TA IoU     =  0.0677
v10 MEASURED TA IoU     =  0.0700
```

So `_07` carried **~1.5× random** worth of real signal, and v10's 5-source
union added **+0.003** on top — inside the noise. Union expansion cannot exceed
the correctness of its best member; it only inflates cardinality and bleeds IoU.
That is exactly what the measurement showed. Ensembling was the wrong lever.

### The one clean structural error: cardinality vs the `tag`

`_07`'s predicted cardinality, split by the `single-answer` / `multiple-answer`
tag that ships in `test.json`:

```
single-answer   (GT size 1):  144 answered with 1 option,  154 answered with 4 options
multiple-answer (GT size ≥2): 152 answered with 1 option,   50 answered with 2–4
```

- **154 single-answer questions were answered with 4 options** → IoU capped at
  0.25 even when correct. Collapsing each to its top-1 pick raises the ceiling to
  1.0 on every one of them.
- **152 multiple-answer questions were answered with 1 option** → guaranteed
  under-recall when GT is ≥2.

That's **306 of 500 questions mis-sized against a label that was handed to us for
free.** Aligning cardinality to the tag is the single highest-leverage,
zero-inference lever, and it was left on the table. `work/track_a_ranker/
phase2_xgb_tuned_shortlist.csv` already emits tag-perfect cardinality
(`{1:298, 4:202}`) plus a within-set ranking (`xgb_top1_score`) — the raw
material for a disciplined submission existed and was never shipped.

**Caveat, stated honestly:** cardinality alignment only pays off if the top-1
pick is the *right* one, i.e. if there's usable within-set confidence ordering.
`_07`'s final sets don't expose a ranking; the XGB ranker does but was never
LB-validated. So the tag-aligned strategy is high-EV, not a guaranteed win —
its upside is bounded by the same ~1.5× content ceiling.

## 3. Track B: 11 right out of 100, and the lift came from 3 questions

Family split: **66 fault + 34 path, 0 topology.** `_07` scored TB = 0.08
(8/100); v10 = 0.11 (11/100). The v10 gain was **3 questions**, from replacing
`_07`'s generic `missing static route` fault fallback (used on ~half the fault
set) with the 7-cluster targeted picks. Real, but tiny in absolute terms.

Why so low: a fault answer must get **both** `node;ip` **and** a reason from a
31-item closed vocabulary exactly right. Random reason alone is 1/31 ≈ 0.032,
and correct `node;ip` multiplies that down further. Getting to 11% means the
agent loop was recovering genuine signal on a handful of questions and guessing
on the rest.

Two facts cap what was achievable offline:

- **Path answers (34) came from the wrong simulator.** The local `server.py` is
  Phase 1's; it returns Phase 1 device names (`DEV-BL-01`, `DEV-PE-01`, …) that
  can never match Phase 2 GT (`Core_SW_01`, `FW_01`, `PE1`, …). Every path
  emission sourced from the local sim was structurally unscoreable. The
  production server (`trackB.organizer.example/api/agent/execute`) returns correct Phase 2
  names and was **never run for a full pass** — credits died first.
- **Fault mode-collapse was untested.** Fresh Qwen picked
  `IP address prefix list missing…` on FW_01 for 53/55 answered fault questions
  (96% concentration). Genuine convergence or collapse was the single biggest
  open question, and the submission that would have tested it (`v13a`) was built
  but never scored.

## 4. What state-of-the-art actually looked like here

Ranked by expected value, given the ceiling is capability-bound:

1. **Real retrieval beats any offline recombination.** The bottleneck was the
   agent tool-loop actually fetching and reasoning over per-question radio /
   network state — not the choice of ensembler. The highest-EV unshipped step
   was a full Track B pass against the **production** tool server (correct Phase
   2 device names, authentic responses). That was blocked only by exhausted
   OpenRouter credit, not by any modeling decision.

2. **Track A: tag-disciplined single-pick, not union.** Ship the XGB ranker's
   top-1 for the 298 `single-answer` questions and a tight (≤2) confident set for
   `multiple-answer`, gated on `xgb_top1_score`. Fall back to the most
   `_07`-aligned source (`results_phase2_fewshot_graph`, a superset of `_07` on
   all 500) only where confidence is low. This directly attacks the 306
   mis-sized questions. Expected TA IoU ceiling ~0.08–0.10 — still bounded by
   content skill, but strictly better than union's 0.070.

3. **Track B: single-line, cluster-targeted, path from production only.** Keep
   v10's fault-cluster picks (the proven +0.03), drop every multi-line emission
   (Accuracy gives them nothing), and take path answers exclusively from the
   production server or from `_07`'s existing fills — never the local sim.

4. **Stop ensembling near-random sources.** With per-source content skill at
   ~1.5× random and highly correlated errors, unions cannot manufacture
   correctness; they only degrade IoU. Every marginal source added after the
   best one was negative EV under this metric.

## 5. The honest ceiling

With the base model fixed at Qwen3.5-35B-A3B and no fine-tuning, the realistic
LB ceiling from *offline recombination alone* was roughly **0.09–0.12** —
squeeze cardinality discipline and precision, but you cannot exceed the model's
content correctness, which measured at ~1.5× random. Breaking past that needed
one of: (a) a working production-server inference pass to convert tool-use into
real per-question signal, (b) fine-tuning on the radio/config task (allowed by
the rules), or (c) a fundamentally better tool-use loop that grounds each answer
in retrieved state instead of guessing. All three were inference-dependent, and
all three were foreclosed the moment the credits ran out on 2026-05-18.

The pipeline wasn't over- or under-engineered on strategy. It was starved of the
one input that mattered: correct, per-question evidence at inference time.

---

## Reproduce the numbers

- Cardinality & agreement: parse `Track A` cells on `|`; `_07` =
  `past_subs/results_07.csv`; sources = `telco_data/Track A/results_phase2_*/result.csv`
  (cols `scenario_id,answers`), `work/track_a_ranker/phase2_xgb_tuned_shortlist.csv`.
- Tags: `telco_data/Track A/data/Phase_2/test.json` → `tag` field (298 single / 202 multiple).
- Random baseline: Monte-Carlo random picks matched to `_07`'s per-question
  cardinality, 24 options, GT size 1 (single) / 2 (multiple) → TA IoU ≈ 0.044.
- Track B: `task.question` family regex (66 fault / 34 path); `_07` fault
  line-count `{1:51, 2:14, 4:1}`; measured anchors `_07`=0.07385, `v10`=0.09002.
