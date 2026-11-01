# Run 5, stages 3 to N: design (V3-058)

Status: approved by the human on 2026-10-30 after two Argon reviews (section 14), with the S3 screen, at most 3
stages and at most 4 advancing slots; built and locked as V3-058 (DECISIONS.md). One change at build time, by the
human: the length limits were raised (fact rules up to 2,500 characters, build check and answering 2,500, example
4,000) and every candidate's length is recorded and shown. Where this document and DECISIONS V3-058 differ in detail,
V3-058 and the code apply.

## 1. Why

- Stage 1 (15 founders + BLIND, 480 generated + 500 hand-written items) found the big gain: FACTS 0.657 against
  plain prose (A0P) 0.503 on the generated items.
- Stage 2 (2^5 factorial on FACTS, 32 designs x 480 items, complete) found that none of the five new parts helps:
  A -0.017, B -0.035, C -0.043 (all Holm-significant, harmful), D -0.003 and E +0.003 (unclear). FACTS unchanged
  (D01) had the highest raw mean, 0.660.
- The locked final-design rule (argmax of a mains + pairs fit) picked D18 (A+, E+), a design that scored lower in raw
  terms, through noisy A:B and A:C interactions. The run was paused before any sealed score was viewed.
- Stages 1-2 showed where the levers are: A, B, C are sensitive (they moved the score 2-4 points); D, E barely move
  it; k=8 is the weak spot (k=1 73%, k=2 65%, k=4 58%, k=8 48%). Failed notes checks rise to 20% at k=8 but failed
  builds score about the same as clean ones at the same k, so the limit is reasoning over large systems, not broken
  notes. Stage 2 spent 15,360 runs on one shot of five ideas.
- The human: run 5 continues with further improvement stages before the trust score is measured. "Tightly
  controlled, measurable evolution": every stage builds on the last and every change is measured.

## 2. Summary of the change

1. **Stage 2 verdict: FACTS stands.** The final-design rule becomes the gated baseline retention rule (section 3).
   Under it, the stage 2 final design is D01 = FACTS. D18 and D17 are dropped.
2. **The opened sealed set is retired.** The final check stops. Its 1,435 records (about 239 each for FACTS, A3, A0P,
   ONT_LOOSE, D18, D17) stay on disk untouched, are flagged `retired_V3_058`, are never read by any analysis, and are
   disclosed (record count only, plus partial means after the final verdict) in an appendix.
3. **Stages 3 to N: screen, then confirm** (sections 4-6), each on fresh generated items.
4. **The final check moves to the end**, on a fresh 720-item sealed set (section 8).
5. **The dashboard gets N stages** (section 10).

## 3. Gated baseline retention rule (applies to stage 2 now and every confirmation factorial)

Start from the stage's current best (D01, the all-minus design). Fit the pre-registered factorial model (mains +
all two-way interactions of the m active factors). Retain main effects that survive Holm at alpha 0.05 with a
positive estimate, and retain two-way interactions that survive Holm at alpha 0.05 together with their two main
effects (marginality); set all other terms to zero; undecided factors default to minus. A factor may not be set to
plus solely through a negative interaction: its own conditional effect at the chosen levels of the other factors
must be positive. The resulting design d* replaces D01 if and only if its direct paired contrast against D01 across
the base systems (mean_d* - mean_D01, cluster t test, df = bases - 1) has its 95% range strictly above zero.
Otherwise D01 stays. A d* that clears this gate with a gain between 0 and +0.005 does become the new best; the
convergence stop (section 6) then ends the stages.

(Review, Argon 2026-10-30: marginality and the no-negative-interaction clause close the D18 failure mode, where
negative A:B and A:C interactions made A+ look good at B-, C-.)

For run 5's stage 2: no term is significantly positive, so D01 = FACTS stays. (FACTS also had the highest raw mean.)

## 4. The five slots (redesigned around k=8)

Each slot edits a separate part of the genome, so any mix of slots builds cleanly. Mapping to FACTS's genes:

| Slot | What it changes | Gene(s) | Aim |
|---|---|---|---|
| 1 Layout | section headers, order, cross-link conventions | `core.params` (headers, granularity) | navigating many connected parts |
| 2 Fact syntax | bullet-level fact templates | `guidance.text`, fact-rules block | unambiguous conditions, states, directed links |
| 3 Build check | how the builder checks its notes before saving (strategy, self-check), not the syntax | `guidance.text`, new separate build-strategy block | no silently dropped cross-part links |
| 4 Answering | how the answerer chains steps across the notes | `answering.text` | multi-hop answers at k=4 and k=8 |
| 5 Worked example | the example's structure and size (e.g. a 2-part chain instead of a trivial one) | `examples.items` | show Luna a multi-part dependency |

Changes in the genome model: `guidance.text` is split into two independent blocks (fact rules, build strategy) so
slots 2 and 3 never collide. Two mechanical invariants, each with a test: (1) FACTS's fact-rules block followed by its
build-strategy block reproduces FACTS's `guidance.text` exactly, so D01 does not drift by a single character; (2) a
slot 3 candidate prescribes only process and verification steps and may not define or quote bullet punctuation,
section header names or checker syntax (fit check). The build budget (old E) is no longer a slot: E did nothing in
stage 2 and doubling it doubles compute.

**Worked example coupling (fixes the B x D contradiction):** the example is always rendered in the design's own layout
and fact syntax. Slot 5 sets the example's content and size; slots 1 and 2 set how it is written. Rendered examples
are indexed by the 3-tuple (slot 1 level, slot 2 level, slot 5 level): if slots 1, 2 and 5 all advance, the
confirmation needs 8 versions, 4 of which exist after the screen and 4 of which Opus writes in an `examples_n` step,
from the advancing texts only (no scores). Opus writes them rather than a code renderer, because a new fact syntax
can be free text that a renderer cannot anticipate. Fit check: every rendered example uses its design's headers and
fact templates, and passes that design's own notes checker with 0 errors.

## 5. Screen (one per stage)

- **Candidates: 20 = 4 per slot.** Opus writes them from a firewalled summary (section 7). The 4 per slot must follow
  4 different archetypes:
  1. Subtract: loosen or remove rules that cause verbosity or needless constraints.
  2. Explicit structure: entity and edge identifiers, input-output tables, aimed at k=8.
  3. Verify: explicit trace or completeness checks before output.
  4. Targeted tweak: keep the current structure and make one surgical change aimed at k=4 and k=8.
- **Diversity check (mechanical, before the screen):** within a slot, the 4 candidates change different rules.
  Archetypes 1-3 differ from each other by at least 25% (token edit distance) within the slot's text; archetype 4
  (targeted tweak) passes if its diff against the current best changes rules or lines disjoint from archetypes 1-3
  and changes at least 10% of the slot.
- **Retry cap:** a candidate (or matched example) that fails any mechanical check goes back to Opus with the
  mechanical reason, automatically, up to 3 times, before the human gate.
- **Design (option S3, second review):** one at a time. 21 designs (the current best + the 20 candidates, each
  changing one slot) on the same 320 fresh generated items (80 base systems x k = 1, 2, 4, 8, balanced 1:1:1:1, same
  family quotas as stages 1-2): 21 x 320 = 6,720 runs (about 5.2 h). Matched worked examples in the screen: a
  candidate for slot 1, 2 or 5 gets its own single-slot example variant (at most 12 per stage, notes checker on each);
  slots 3 and 4 keep the current example.
- **Precision (measured, not assumed):** the cluster SE of one paired design-vs-design difference on stage 2's records
  is 0.0125 on 480 items (120 bases), 0.022 on 160 (40 bases), so about **0.0153 on 320 items (80 bases)**. Rejected
  on review: S1 (an L25 orthogonal array; 25 multi-slot examples before the screen, and one broken candidate would
  confound the other slots' estimates) and S2 (21 x 480 = 10,080 runs per screen).

- **Advancement gate (per slot):** take the slot's best candidate c* by its overall paired difference against the
  current best. It advances if and only if the difference is at least +0.015 (t >= 1.00 at SE 0.0153). There is no
  separate k=8 gate (80 items at k=8 give SE about 0.03, too noisy to gate on); k=8 is reported, not gated. Slots
  with no advancing candidate stay at the current best.
- **Operating figures (simulated: best of 4 per slot, comparisons sharing the current best, SE 0.0153):** under the
  null a slot advances 37% of the time (about 1.9 of 5 slots; no slot advances 10% of the time); a single real
  candidate advances with probability 0.42 at +0.015, 0.69 at +0.025, 0.89 at +0.035. The screen only filters: the
  confirmation factorial on 480 fresh items decides.
- **At most 4 slots advance (proposed, human to confirm):** if 5 pass the gate, the 4 with the largest differences
  go on (5 passing by chance alone happens under 1% of the time). This caps the confirmation at 2^4 = 16 designs.
- **Futility stop:** if no slot advances, the stage ends with the current best unchanged and the stages stop.

## 6. Confirm (one per stage)

- **Factorial 2^m** over the m advancing slots (m = 1 to 4); failing slots frozen at the current best. If m = 1, a
  direct paired test of the one candidate against the current best.
- **Items:** 480 fresh generated items (120 bases x 4 k), disjoint from the screen's.
- **Runs:** 2^m x 480: at most 7,680 (m = 4, about 6 h), 3,840 (m = 3), 1,920 (m = 2), 960 (m = 1).
- **D01 is the current best**, run fresh on the confirmation items.
- **Analysis:** primary paired cluster tests (as in stage 2); Holm family = all m mains + all m(m-1)/2 pairs;
  selection by the gated baseline retention rule (section 3).
- **Convergence stop:** the stages stop if no design replaces D01, or the confirmed gain over D01 is under +0.005.

## 7. What Opus sees (firewall, V3-056 rules extended)

Allowed in the summary for stage n: the current best's genome; its score overall and by k on the previous stage's
confirmation items; notes-check failures by checker category and by k; tool calls and tokens by k; how the build
tools behave (read from the design); every earlier stage's candidate texts with their measured effects and ranges
(which helped, hurt or were unclear): full text for advancing candidates and the worst candidate per slot, a
compact table (slot, archetype, one-line description, effect, range) for the rest, so the summary stays small by
stages 4-5; the B x D lesson (examples must match the fact syntax).
Never: hand-written items or results, scores by problem family, any item, question, answer key, or anything from any
sealed set.

## 8. Final check (once, at the end of all stages)

- **A fresh 720-item sealed set** (180 bases x 4 k), never used before.
- **Designs:** the final design (FACTS*), FACTS, A0P (plain prose), A3 (runner-up), ONT_LOOSE (hand-written top).
  If FACTS* = FACTS, it runs once.
- **Fixed-sequence tests at alpha 0.05, each counted only if all earlier ones pass, largest effect first** (review:
  a small real gain over FACTS that misses p < 0.05 must not block the headline):
  1. FACTS* vs A0P: the headline against plain prose.
  2. FACTS* vs ONT_LOOSE: against the hand-written top founder.
  3. FACTS* vs A3: against the stage 1 runner-up.
  4. FACTS* vs FACTS, only if they differ (otherwise not applicable): did the stages improve FACTS?
  If FACTS* = FACTS, tests 1-3 run on FACTS and the logic is unchanged. Note: on 720 items (180 bases) the SE of a
  pairwise difference is about 0.010 (scaled from the measured 0.0125 at 120 bases), so a real gain under about
  +0.02 may not reach p < 0.05 in test 4; it is then reported with its range, not claimed.
  Also reported, not gating: FACTS vs A0P, scores by k.
- **Hand-written items:** FACTS* on the 500 hand-written items (stage 1's records for the others are reused);
  FACTS* vs FACTS there as the out-of-distribution check, a paired test across the 500 items (df 499), unlike the
  cluster test across base systems used on generated items. These items were used in stage 1 only to name the
  hand-written top founder and were never shown to any designer.
- **Runs:** 5 x 720 + 500 = 4,100 (about 3.2 h); 4 x 720 = 2,880 if FACTS* = FACTS.
- **Trust report:** final after this check, with every stage's changes (kept or not) and their before and after.

## 9. Maximum stages, timeline and cost

- **Stage cap (review recommendation):** at most 3 improvement stages (stages 3, 4 and 5), plus the futility and
  convergence stops. A finite cap lets every item set (3 x (80 screen + 120 confirmation) bases + a 180-base sealed
  set = 780 bases) be allocated, checked for overlap and hash-locked once, before stage 3 starts. The code supports
  any N; the cap is a pre-registered number. Human to confirm 3.
- **Per stage:** Opus writes candidates and their single-slot examples (up to 30 min) -> human gate -> screen
  (~5.2 h) -> Opus writes matched examples for the confirmation cube (short) -> human gate -> confirmation (up to
  ~6 h). About 12 h per stage at most.
- **Three stages + final check, worst case:** 3 x (6,720 + 7,680) + 4,100 = 47,300 runs (~36 h). The stops usually
  end it sooner, and a confirmation with m < 4 is much smaller (m = 2: 1,920).
- **Cost:** generated items and their grading are code (cost 0). Hand-written grading is DeepSeek, about $0.0012 a
  call: about $0.60 for the final check. Luna runs come from the ChatGPT Pro allowance (weekly guard at 90%); Opus
  calls from the Claude plan (guard at 85%).

## 10. Human gates (mechanical only)

Before each screen and each confirmation the human checks only: (a) firewall and safety compliance; (b) slot
isolation (a slot's candidate changes only its own block, plus its matched example lines); (c) length limits. Any
candidate that passes enters untouched: no wording edits, no rejections by taste. A failed check returns the
candidate to Opus with the mechanical reason (as PLUS_REJECT does now).

## 11. Dashboard: N stages

- The stage list comes from the run's state, not hard-coded: Stage 1, Stage 2, then Stage 3 screen and Stage 3
  confirm, ..., Final check. Each gets a tab and a progress tile with its run count.
- Screen tab: the 20 candidates by slot, their lead over the current best overall and at k=8, which advanced.
- Confirm tab: the same panels as Stage 2 (parts, effects, designs).
- Interaction panel renamed "Interaction terms (synergy / antagonism)", each with its design's score and its
  difference from D01, so a less-bad pair is never read as a winner.
- Trust report: every stage's changes with before and after; the D18/D17 retirement disclosed.

## 12. Fresh items: the open problem

Generated items are seeded code: each base system uses a window of 500 seeds, and every new set must be disjoint from
all earlier ones (checked by `scripts/check_item_ranges.py`). Three stages plus a new sealed set need 3 x (80 + 120)
+ 180 = 780 new bases (to verify in code before building: the unchosen windows in the reserved main, sealed and pilot
ranges are about 945 at most, before any the scan rejected). Unused contiguous seed space is tight (about 54k at the top of the validation range, 29k in
train). Options:
- (a) use the windows inside already-reserved ranges that were scanned but never chosen (never run, never seen;
  the checker confirms no overlap);
- (b) lower the scan factor (4n slots per n bases) for the new sets;
- (c) a smaller confirmation set (e.g. 360 items).
Proposed (accepted in review): (a) plus (b), with the checker as the gate; the confirmation set stays at 480 items
(not (c)). Conditions: the earlier scans chose bases round-robin by problem family, not by content, so the unchosen
windows were never run or seen (to verify in code before use); new sets keep the same family and k quotas as stages
1-2; `scripts/check_item_ranges.py` must show zero overlap at the 500-seed window level against every active and
retired set (including run 5's retired sealed set and runs 1-4). Also check whether the split bounds can simply be
extended into unused seed space.

## 12b. Audit records before restarting (review)

1. The retired sealed records: a manifest (path + SHA-256 of each of the 1,435 record files, plus one hash of the
   manifest) stored with DECISIONS V3-058, so anyone can verify they were not changed or extended after 12:25 on
   2026-10-30.
2. The new 720-item sealed set (`sealed_v2`) and every stage's item sets are allocated, checked and recorded (seed
   ranges + content hash) in PREREG before Opus is called for stage 3.
3. A test that `v3/analysis5.py` and `v3/plus_evidence5.py` refuse (raise) on any record flagged `retired_V3_058`.

## 13. What gets built (after approval)

- `v3/run5.py`: table-driven stage machine (stage1, plus, stage2, screen_n, examples_n, confirm_n, ..., final);
  the final check after the last stage; per-stage record folders (`runs/run5_s3a`, `run5_s3b`, ...).
- `v3/analysis5.py`: gated retention rule; screen ranking and advancement gate; 2^m fit with the m + m(m-1)/2 Holm
  family; fixed-sequence final tests.
- `v3/plus5.py` -> a stage writer: 20 archetype candidates, diversity check, matched examples, fit check.
- `v3/plus_evidence5.py`: the stage n summary (section 7).
- `v3/run5_sets.py`: new disjoint item sets and the new sealed set.
- Retire the old sealed set (flag only; records untouched).
- Dashboard N stages (section 11).
- Tests for each, including one that fails on the old final-design rule; full suite; DECISIONS V3-058 and PREREG
  entries; re-lock; backup; restart through the runner.

## 14. Review log

- 2026-10-30, Argon, first review: required changes accepted and written in (marginality in the retention rule,
  tighter screen gate, 3-tuple example cube with notes-checker pass, cap of 3 stages, final tests reordered largest
  effect first); optional changes accepted (diversity nuance for archetype 4, compact summary of past candidates);
  implementation details accepted (retry cap of 3, df 499 paired test on hand-written items, audit records 12b).
  One review number corrected from our data: the 160-item screen SE is 0.022, not 0.014; this reopens the screen
  design (section 5, options S1 and S2).
- 2026-10-30, Argon, second review: S1 and S2 rejected, S3 adopted (one at a time on 320 items, gate +0.015 at
  t >= 1, no k=8 gate); sections 3, 4, 8, 12 and 12b approved unchanged. Corrected from our simulation: the null
  per-slot advance rate is 37%, not 28.5% (the four comparisons share the current best); power at +0.025 is 0.69,
  not 0.74. Review assumed a cap of 4 confirmation slots that the draft did not have: now proposed in section 5.
  Worst case 47,300 runs (the review's opening figure of 42,600 did not match its own sum).

## 15. Questions for review (first round)

1. Is the gated baseline retention rule, applied to stage 2 now, sound as written in section 3?
2. Is retiring the opened sealed set (records kept, unviewed, disclosed) and using a fresh one at the end enough to
   meet your earlier objection to Option B?
3. Slots: is splitting `guidance.text` into fact rules (slot 2) and build strategy (slot 3) clean enough, and is
   dropping the build budget as a slot right?
4. Matched examples written by Opus after the screen (from the advancing texts, no scores): acceptable, or should a
   deterministic renderer do it?
5. Stage cap: 2, 3, or N with only the stopping rules?
6. Fresh items: (a) + (b) in section 12, or another way?
7. Final check: are the fixed-sequence order and the designs in section 8 right?
