# Run 5: a designed experiment in two stages (DRAFT for the human's approval, 2026-10-28)

Status: draft. Nothing is built or locked. Run 4 is archived (`archive/run4_2026-10-28/`, V3-041) and the white paper
is on hold; it will be rewritten around run 5 "as if run 4 never existed", keeping its format.

## 0. Revisions after the independent review (V3-044, approved by the human 2026-10-29)

**Where this section differs from the text below, this section applies.** Full list and reasons: DECISIONS V3-044.

| Topic | Now |
| --- | --- |
| Main set | 120 bases x 4 k = **480** generated items (was 360) |
| Founders | the 15, plus a **BLIND** baseline (no description, no notes; answers from the question alone), never chosen |
| Failures | caused by the design (forbidden tool call, out of time): retried twice, then **scored 0**; infrastructure: retried with backoff, then the run **halts** for the human |
| Stage 1 winner | highest generated mean; ties broken by the hand-written mean, then name. The hand-written top founder joins the final check if it is neither winner nor runner-up |
| Stage 1 extra | pre-registered contrast: notes-only founders vs tool-assisted founders |
| Stage 2 | all **32 designs run fresh** (design 1 does not reuse stage 1 records); primary tests **across base systems**; pre-registered: A, B, C, D, E, A x B, A x C, B x C, A x B x C (**nine**, Holm); the 16-df fit is secondary; final design = argmax of mains + pairs (+ A x B x C if p < 0.05) |
| + texts | the human may send them back only for a written reason, at most twice; B's length 0.5 to 3 times the winner's (at least 600 characters allowed) |
| Final check | in order: (1) stage 1 winner vs control; (2) final design vs winner, if they differ. If the final design has the doubled budget, the winner with only the doubled budget also runs, and the lead splits into recipe and compute |
| Hand-written items | leak check: no-document score at most 0.25 and no hit on the main key point; 60 top-up systems with a diagnosis question; the 500 chosen by quota per question type |
| Judge | 100 stage 1 hand-written answers re-graded by Claude Opus (reported, not gating) |
| Factor A | for a notes core (sectioned, ontology, JSON, Markdown) the layout parameters; for any other core a layout paragraph added to the guidance (V3-048: a schema change never reached one-shot builders); the 32 designs must render to 32 distinct guides / tool sets |
| Secondary outcomes | output tokens and wall-clock time per design; accuracy conditional on a passed format check |

**Precision (corrected from the pilot):** paired sd 0.36, within-base correlation 0.03 (design effect about 1.1).
80%-power leads: about 0.049 for two founders head to head on 480 items, about 0.040 in the final check on 720;
one stage 2 part about 0.014 to 0.017.

**Budget:** stage 1 16 x 980 = about 15,700 (BLIND is answer-only); stage 2 32 x 480 = 15,360; final check 4 to 6
designs x 720 plus up to 500 = about 3,400 to 4,800. Total about 35,000 runs, about 35 hours at the pilot's pace,
spread over about one and a quarter weeks of Luna allowance.

## 1. Why run 5 changes (the human, 2026-10-28)
- "relying on chance to pick the winner is silly": run 4 bred children by chance and sorted them through stages
  (quick check, screens at 35 and 65 items, full test, growing proof). The stages cut designs on luck, gave designs
  different numbers of items, and every stage added a winner's curse.
- "just run everything through 340 tests or whatever, 20 at a time": every design meets the same items, once, with 20
  Luna runs at a time. No evolution, no screens, no special breeds.
- "first principles": run 5 starts from the founding designs, not from run 4's champion, and every new version of a
  part is written without run 4's results.
- "i also want to include markdown and JSON structure": two new founders.

## 2. The two stages in one paragraph each
**Stage 1, which way of describing reality works best.** The fifteen founding designs (nine from before, the causal graph A2,
the simulation A3, the probability model G1C1, plus JSON, Markdown and Luna's own choice) are each a different way of
writing a system down. All fifteen are tested on the same 360 generated items and 500 hand-written-style items.
Because every founder answers the same items, they are compared item by item, and the result is a ranking with honest
ranges: which formats are clearly better, and which are tied.

**Stage 2, what makes the winner better.** The stage 1 winner is broken into five parts (how its notes are laid out,
how facts are written, how it answers, whether it shows a worked example, how long it may think). For each part one
alternative version is written from first principles. All 32 combinations of the five parts are built, so each alternative is
switched on in exactly half of them, balanced against every other part, and all 32 run on the same 360 items. Each
part's effect is then measured by all 32 designs at once. The best combination gets one honest re-test on sealed fresh
items.

## 3. Stage 1: the founders

| Founder | Way of describing reality | Core | Group |
| --- | --- | --- | --- |
| A0P (control) | plain prose, no structure | prose | notes only |
| SEC_A1 | sections: entities, state, mechanisms, causal links, timing, uncertainty (tagged) | sectioned | notes only |
| ONT_LOOSE | an ontology: things, hierarchy, relationships, rules | ontology | notes only |
| SEC_FLOW | purpose, parts, flows, rules, timeline, unknowns | sectioned | notes only |
| G1C3 | equations | equations | fixed calculator |
| A5 | rules | rules | fixed calculator |
| FACTS | short facts: things, values, rules, links, timing | sectioned | notes only |
| STATES | parts, states, events, transitions, timing | sectioned | notes only |
| SPEC | a specification: context, components, interfaces, requirements, constraints | sectioned | notes only |
| A2 (added) | a causal graph: variables, cause-and-effect arrows, equations | graph | fixed calculator |
| A3 (added) | a simulation: Luna writes a program that plays the system forward | simulation | own code |
| G1C1 (added) | a probability model: Luna writes code for the uncertain parts | probabilistic | own code |
| **JSON** (new) | one JSON object, keys chosen by the founder's own schema | new, see below | notes only |
| **MARKDOWN** (new) | Markdown headings with tables for things, values and links, and rule bullets | new, see below | notes only |
| **OWN** (new) | Luna's own choice: no format suggested | prose | notes only |

The nine, A2, A3 and G1C1 are unchanged from their seed files (`v3/genome/seeds/`). The two new founders are written from first
principles at the same level of effort as the others (not by wrapping FACTS): a design brief per format, written and
shown to the human before the lock. Decided (human, 2026-10-28): real cores, so the harness checks the format the way it
checks the sectioned core (JSON: must parse as one object, one repair turn if it does not; Markdown: tables must have
consistent columns), and record the share of invalid notes. The 2026-10-26 format check (5 items, prose core with a
format instruction, `paper/side/format_check.json`) was too small to tell anything and is not used.

**OWN (human, 2026-10-28).** The baseline question: do designed formats beat what the model does on its own? Neutral
instruction, for example "Write notes about this system in whatever form you judge best for answering questions about
it later", with no hint of sections, JSON or tables; plain notes core, no format check. The harness records, by code
only, the form Luna chose on each item (valid JSON, Markdown tables or headings, bullets, prose, code-like), so the
run also reports which format Luna reaches for, whether it changes with system size, and whether it relates to the
score. If OWN wins stage 1, factor A of stage 2 becomes "a suggested structure" versus "none".

Runs: 15 founders x 360 items = 5,400.

**Three groups, reported separately as well as together (human, 2026-10-28).** Tools stay as they are in each core.
- Notes only: the tools only store and search the notes (lookup, path, hierarchy).
- Fixed calculator: Luna writes the model (equations, rules, a causal graph); a fixed program the experiment provides
  calculates with it (evaluate, intervene, run_scenario).
- Own code: Luna writes Python that models the system and runs it. The human: "it has to understand the english to
  write the code". Caveat for the paper: a description that needs code to be useful carries over less well to
  settings where an AI can only read text.

**Own code runs in Docker (human, 2026-10-28: "yes use docker").** Lifting the Codex refusal of these two cores
(`REFUSED_CORES`, V3-011) is a safety-rule change, recorded as a decision. Every program Luna writes runs in a
throwaway container: no network (`--network none`), read-only system, a small scratch folder only, memory, CPU and
time limits, not as an administrator, removed after the call. The v2 audit-hook sandbox
(`rpp-core/harness/sandbox_runner.py`) stays inside the container as a second layer. Before the lock, attack tests
must all fail: reach the internet, read a planted file outside the folder, write to the host disk, start a
program, fork without limit, use 10 GB of memory, run forever. Docker Desktop must be running during the run.

Not included: ONT_BASIC (dropped by the human for run 3; ONT_LOOSE covers ontologies). A2, A3 and G1C1 were never
tested by Luna (A3 and G1C1 were refused for safety; A2 has no recorded reason), so they are added.

## 4. Stage 2: five parts of the winner, varied by design

**Choosing the winner (decided, human, 2026-10-28).** The founder with the highest mean in stage 1 goes forward,
whether or not it is tied with others (section 6.1); a tie is reported openly. The runner-up (second-highest mean)
also meets the final design in the final check (section 6.4).

**The parts (factors).** Level "-" is always the winning founder exactly as it is. Level "+" is one alternative,
written from first principles:

| Factor | What it is | "+" |
| --- | --- | --- |
| A notes layout | how the notes are organised (sections, keys, columns) | an alternative organisation of the same content |
| B fact writing | the rules for writing each fact down | an alternative set of writing rules |
| C answering | the procedure for answering from the notes | a step-by-step answering procedure |
| D worked example | an example of good notes | none / one short example (whichever the founder lacks) |
| E thinking budget | max_turns | the founder's value / double it |

**First principles, concretely.** Claude Opus writes each "+" version seeing only: the winning founder's genome, the
task description (describe a system so that a small model can answer questions about it), the factor's definition
and the fit rule below. It never sees run 4 (no scores, designs or lessons), any stage 1 score beyond which founder
won, or any item, question or answer key. All five texts are written before stage 2 starts, frozen, and shown to the
human. Fit rule (a code check): every combination of levels must render; no answering text may name a section, key
or column that a layout lacks.

**The 32 designs: the full 2^5 factorial (decided, human, 2026-10-28: "just do 32").** Every combination of the
five parts, in standard order:

| # | A | B | C | D | E | | # | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | - | - | - | - | - | | 17 | - | - | - | - | + |
| 2 | + | - | - | - | - | | 18 | + | - | - | - | + |
| 3 | - | + | - | - | - | | 19 | - | + | - | - | + |
| 4 | + | + | - | - | - | | 20 | + | + | - | - | + |
| 5 | - | - | + | - | - | | 21 | - | - | + | - | + |
| 6 | + | - | + | - | - | | 22 | + | - | + | - | + |
| 7 | - | + | + | - | - | | 23 | - | + | + | - | + |
| 8 | + | + | + | - | - | | 24 | + | + | + | - | + |
| 9 | - | - | - | + | - | | 25 | - | - | - | + | + |
| 10 | + | - | - | + | - | | 26 | + | - | - | + | + |
| 11 | - | + | - | + | - | | 27 | - | + | - | + | + |
| 12 | + | + | - | + | - | | 28 | + | + | - | + | + |
| 13 | - | - | + | + | - | | 29 | - | - | + | + | + |
| 14 | + | - | + | + | - | | 30 | + | - | + | + | + |
| 15 | - | + | + | + | - | | 31 | - | + | + | + | + |
| 16 | + | + | + | + | - | | 32 | + | + | + | + | + |

Design 1 is the stage 1 winner itself: its stage 1 records are reused (same items), so stage 2 runs 31 new designs.
Each column has sixteen "+" and sixteen "-", and any two columns agree in exactly half the rows (orthogonal). With
every combination run, every effect (each part, every pair, and the higher interactions) is measured separately,
with no assumption that any of them is zero.

Runs: 31 x 360 = 11,160.

## 5. Items
- One main set of 360 fresh generated items (90 bases x system sizes k = 1, 2, 4, 8; problem families balanced round
  robin) from a range no earlier run used (checked by `scripts/check_item_ranges.py`). Stage 1 and stage 2 use the
  same set, so every comparison in run 5 is paired.
- One sealed final set of 720 more fresh items, opened once for the final check.
- **500 new hand-written-style items (decided, human, 2026-10-28), reported separately.** Like the v2 hand-written
  items: a long, messy, realistic document (logs, emails, meeting notes; about 6,000 characters), one open question
  (prediction, diagnosis, intervention, assumptions, completeness, consistency) and an answer key (summary, key
  points, typical wrong answers), graded by the AI judge (DeepSeek v4.1 Flash, as before). About 170 invented
  systems x 3 questions, balanced over domains, kinds (continuous, discrete, stochastic) and messiness (realistic,
  adversarial), following the v2 construction protocol: the system is designed first, then the document is written
  about it, so the answer key is known exactly.
  - Written by Claude Sonnet in separate writer agents (not an OpenAI model, so not tuned to Luna's family). Writers
    never see a design; the Opus agents writing the stage 2 "+" versions never see these items.
  - Checked before the lock, and any item that fails is replaced: (1) a second Sonnet agent, given the document,
    must reach the key, and confirms the key agrees with itself and the document; (2) DeepSeek, given the question
    without the document, must fail (catches give-away questions and things the models already know); (3) the human
    spot-checks about 20. The paper says plainly that these are AI-written items in a hand-written style.
  - The 27 v2 dev items are not used (earlier runs saw them); the v2 held-out split stays unopened.
  - Used in stage 1 (all 15 founders) and the final check, never to choose anything: the stage 1 winner, the
    runner-up and the final design are chosen on the generated items only. So the founders' hand-written records
    can be reused in the final check, and only the final design needs new runs on them.
- Every design meets every main item exactly once. A failed run is retried twice and reported, never scored 0.

## 6. The mathematics
Notation: design d, item i (n = 360), y(d,i) = the share of item i's questions design d answers right.
Measured in run 4's data (`paper/side/run5_noise/noise.json`): item noise sigma = 0.18 per item among good designs
(0.22 across all), and the difference between two designs on the same item varies with sd 0.256.

### 6.1 Stage 1: comparing founders on the same items
Model: y(d,i) = mu + a_i + delta_d + e(d,i). a_i is the item's difficulty and cancels in every comparison, because
both founders answered the same item: the comparison of founders d and d' is the average over items of
y(d,i) - y(d',i), with standard error sd(difference) / sqrt(n) = 0.256 / sqrt(360) = 0.0135 (a 95% range of about
+-0.026). Plain English: two founders are compared like two students sitting the same exam, question by question, so
an easy or hard exam cannot favour either.

Which founders could be best: multiple comparisons with the best (Hsu's method). Founder d stays in the "could be best"
set unless the best other founder beats it by more than a margin that holds for all fifteen at once (computed by
bootstrapping items: resample the 360 items many times, recompute every founder's mean, and take the 95th percentile
of the largest gap that luck alone produces). Plain English: we only say a founder is out of the running if the gap is
bigger than the biggest gap luck could make among fifteen founders, not just between two.

Also reported: the three groups (mean of their founders), every founder's lead over the control, and every founder by system size k and by problem family, each
with a 95% range (bootstrap over items). Holm correction for the fourteen "better than the control" tests.

### 6.2 Stage 2: the factorial
Model: y(d,i) = mu + a_i + sum over factors f of beta_f x_f(d) + sum over pairs (f,g) of gamma_fg x_f(d) x_g(d)
+ u_d + e(d,i), with x = -1 or +1. u_d is design surprise (what the parts and pairs do not explain, including the
three-, four- and five-part interactions; variance tau^2), e is item noise (sigma^2). Two pairs are named in advance
as confirmatory because both concern the answering procedure, which reads the notes the layout (A) and the fact
writing (B) produce: A x C and B x C.

The estimate is a difference of averages: effect of f = (mean of the 16 designs with f = "+") - (mean of the 16 with
f = "-"). Everything else is balanced across the two halves, so it cancels. Plain English: each part is judged by all
32 designs, half with it and half without, so no other part can pass for its effect.

Precision: Var(effect) = 4 (sigma^2 / n + tau^2) / N, with N = 32 designs and n = 360 items.

| | standard error | 95% range | smallest effect found 80% of the time |
| --- | --- | --- | --- |
| Two designs head to head on 360 items | 0.0135 | +-0.026 | 0.038 |
| One part across 32 balanced designs, tau = 0 | 0.0034 | +-0.007 | 0.009 |
| Same, tau = 0.01 (planning value) | 0.0049 | +-0.010 | 0.014 |
| Same, tau = 0.02 | 0.0078 | +-0.015 | 0.022 |
Plain English: each part is measured about three times more sharply than two designs compared head to head, because
every run counts towards every part. Design surprise (tau) is not reduced by more items, only by more designs, which
is why it is 32 designs on 360 items and not eight designs on 1,440.

Testing: the model is fitted to the 32 design means (item effects cancel): intercept, five parts and all ten pairs =
16 terms, leaving 16 degrees of freedom (the three-, four- and five-part interactions) whose scatter measures
tau^2 + sigma^2/n in this run itself, so the t-tests (16 degrees of freedom) need no outside numbers. Seven
pre-registered tests (five parts, A x C, B x C), Holm-corrected (the chance of any false "yes" stays at 5%). The
other eight pairs are reported with their ranges as exploratory. The higher interactions are also shown on a
half-normal plot (Lenth's method): if any stands out, it is reported and the noise estimate is redone without it.
Effects by k and by family are reported with ranges, descriptive only.

### 6.3 The final design and the winner's curse
The final design is the combination with the highest score fitted by the model (five parts and ten pairs), not the
highest observed score. Every combination was run, so it is one of the 32. Its fitted score rests on effects averaged
over all 32 designs, so one lucky design cannot make it; in contrast, the highest of 32 observed scores would be
inflated by luck by about 2.07 x sqrt(sigma^2/n + tau^2) = +0.029 at the planning values.

### 6.4 The final check
The final design, the stage 1 winner, the stage 1 runner-up and the control, on the 720 sealed items, opened once.
Pre-registered: the final design is declared better than the stage 1 winner (and, separately, than the runner-up) if the 95% range of its paired lead stays above zero; with 720 items
that needs a lead of about 0.256 x 1.96 / sqrt(720) = 0.019. Plain English: one honest re-test of the final answer, on
items nobody has seen, so the result cannot be luck.

### 6.4b Hand-written items (second, separate test)
Stage 1: each founder's mean on the 500 hand-written items with its 95% range, the same comparison with the best
(Hsu) as section 6.1, reported beside the generated-item results, and the rank agreement between the two (Spearman)
with a range. It does not change the stage 1 winner. Final check: the final design against the stage 1 winner and the
runner-up on the 500, the same pass rule as section 6.4; with a paired spread of about 0.30 that needs a lead of about
0.30 x 1.96 / sqrt(500) = 0.026. Pre-registered as a second verdict: the paper reports both, and says where they
disagree. Plain English: the generated items say which design is best at the kind of question we can mark exactly;
the hand-written items say whether that still holds on realistic messy text.

### 6.5 Pilot
8 designs (FACTS, JSON, MARKDOWN, OWN, A2, A3, G1C1, A0P) x 60 items = 480 runs, to check that the new cores, A2 and the
own-code founders in Docker (never run on Luna)
build and parse, and that
sigma is as expected, before the lock. Its records are not used in the results.

## 7. Budget and time (GPT-6 Luna, Codex, 20 concurrent)
| Part | Runs |
| --- | --- |
| Pilot | 480 |
| Stage 1: 15 founders x 360 | 5,400 |
| Stage 1 hand-written: 15 founders x 500 | 7,500 |
| Stage 2: 31 new designs x 360 | 11,160 |
| Final check: 4 designs x 720 | 2,880 |
| Final check hand-written: final design x 500 | 500 |
| Total | about 27,900 |
At run 4's pace (about 350 runs an hour) about 80 hours of running, likely spread over two weeks by the weekly Codex
allowance (the guard pauses at 90%). Judge cost for the hand-written items: about 8,000 x $0.0012 = about $10
(run 4 ledger: 754 judge calls, $0.0012 each); the no-context checks about $0.60. Writing and checking the items
runs on the human's Claude plan, not a pay-per-call API. Stage 1 is a natural break: it is a complete result on its own.

## 8. Dashboard (streamlined, no creature pictures)
One page. Stage 1: each founder's score so far with its range, the "could be best" set, leads over the control, by k
and family. Stage 2: the 32 designs x five parts table with scores, the five effects and ten pairs with ranges
narrowing as data arrives, sigma and tau. Always: progress (runs done of planned), rate, spend and the weekly
allowance. Names are plain text labels (founder names and design numbers).

## 9. What is fixed before any run (pre-registration)
The founders and the two new founder designs; the item ranges; the stage 1 analysis and the rule for choosing the
winner; the five factors, the brief Opus gets for the "+" versions, and the fit rule; the 32-design layout; the 500 hand-written items, their checks and their analysis; the model,
the seven tests and Holm; the final-design rule; the final check and its pass rule; the pilot. Then lock and verify,
as before. Anything changed after the lock is a recorded deviation.

## 10. To decide before locking
1. DECIDED (2026-10-28): the nine, A2, A3, G1C1, JSON, MARKDOWN and OWN (15 founders); tools as they are, three groups;
   own code in Docker.
2. DECIDED (2026-10-28): JSON and MARKDOWN as real checked cores (format check, one repair turn, lookup tools).
3. DECIDED (2026-10-28): the first-ranked founder goes to stage 2; the runner-up joins the final check.
4. DECIDED (2026-10-28): the full 32 designs in stage 2.
5. DECIDED (2026-10-28): final check on 720 sealed items.
6. DECIDED (2026-10-28): 500 new hand-written-style items written and checked by Claude Sonnet, in stage 1 and the
   final check, reported separately (section 6.4b).
