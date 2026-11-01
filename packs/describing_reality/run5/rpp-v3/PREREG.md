# Discovery Programme v3: pre-registration

Frozen by `state/PREREG.LOCK` before generation 1 (`python -m v3.prereg lock --reason ...`). Any change after the lock
needs a new, recorded lock with its reason, and the final report lists every lock.

## Run 5 (V3-042 to V3-048): read this first
Run 4 was stopped and archived by the human on 2026-10-28 (V3-041, `archive/run4_2026-10-28/`). **Run 5 is a new
experiment: a designed experiment in two stages, not evolution. Its rules are `docs/RUN5_DESIGN.md` (locked with
this file; its section 0 lists the V3-044 revisions after an independent review) and the summary below; where anything
later in this file differs, the run 5 rules apply.** The sections for runs 1-4 are kept unchanged as their record.

Locked for run 5:
- Builder: GPT-6 Luna through Codex (allowlisted tools; any other call is a design-induced failure), the two-phase
  episode, 20 runs at a time (`v3/run5.py`). Every (design, item) pair runs once per record folder. A failure caused
  by the design (forbidden tool call, out of time) is retried twice, then scored 0; an infrastructure failure is
  retried with backoff, then halts the run for the human (never dropped).
- Items (`v3/run5_sets.py`, ids in the lock): main 480 generated (120 bases x k = 1, 2, 4, 8); sealed 720 generated,
  opened once at the final check (`state/run5/opened.json`); 500 hand-written-style items (`items/hand5/items.json`,
  hash in the lock; leak check answered by Claude Sonnet without the document, <= 0.25 with no hit on the main key
  point; chosen by quota per question type), graded
  by DeepSeek v4.1 Flash. No overlap with any earlier run (`scripts/check_item_ranges.py --run5`). V3-051: the final
  items carry kind / domain / messiness from their system record and have the writer tool server's cp1252 mojibake
  undone exactly (`v3.hand5.factory.unmangle`); the tool server now reads UTF-8.
- Stage 1: the 15 founders (A0P control, SEC_A1, ONT_LOOSE, SEC_FLOW, G1C3, A5, FACTS, STATES, SPEC, A2, A3, G1C1,
  JSON, MARKDOWN, OWN) and the BLIND baseline (never chosen) on the 480 and the 500. Stage 1 tests representations
  together with their tools. Winner and runner-up: the highest means on the generated items (ties: the hand-written
  mean, then name). Could-be-best set: Hsu, studentised and one-sided (cluster bootstrap); leads over A0P with Holm
  (14 tests, percentile-bootstrap p);
  pre-registered contrast notes-only vs tool-assisted. Judge calibration: 100 hand-written answers re-graded by
  Claude Opus (reported).
- Stage 2: the full 2^5 factorial on the winner's five parts (A notes layout: the core's layout parameters for a
  notes core, otherwise a layout paragraph added to the guidance; B fact writing, C answering, D worked example,
  E thinking budget), the 32 designs rendering to 32 distinct guides / tool sets, "+" texts written by Claude Opus (`v3/plus5.py`; the winner genome
  and, from V3-056, a summary of the winner's stage 1 evidence on the generated items only (`v3/plus_evidence5.py`:
  score overall and by k, notes-check failures by checker category and by k, build tool calls by k, output tokens,
  and from V3-057 how the build tools behave, read from the design; never anything
  from the hand-written items, scores by family, the pass-only score, or any item, question or answer key); the example in both layouts; B and C name no section or key; A+ keeps names B- uses), fit-checked (including
  B's length rule and 32 distinct designs), frozen with the winner in the hash, and approved by the human (sent back only for a written
  reason, at most twice) before any stage 2 run. All 32 designs x the 480, run fresh (`runs/run5_s2`). Primary tests
  across base systems; nine pre-registered (A, B, C, D, E, A:B, A:C, B:C, A:B:C) with Holm; the 16-df fit on the 32
  means and a Lenth check are secondary.
- Final design: the argmax of the fitted model (five mains and ten pairs, plus A:B:C if p < 0.05), not the highest
  observed. Final check on the sealed 720, in order: (1) stage 1 winner vs control; (2) final design vs winner, if
  they differ. Also vs the runner-up, the hand-written top founder (if neither winner nor runner-up), the control,
  and, when the final design has the doubled budget, the recipe / compute split. The same comparisons on the 500
  (founders' records reused). Pass = the 95% cluster-bootstrap range of the paired lead above zero.
- Secondary outcomes: output tokens and wall-clock time per design; accuracy conditional on a passed format check.
- Own code (A3, G1C1) runs only in the Docker sandbox (`v3/sandbox.py`, `sandbox/`; V3-043); attack tests pass.
- All analysis is `v3/analysis5.py` (code only, fixed bootstrap seed 20261028, 10,000 resamples), tested on
  simulated data (`tests/test_analysis5.py`).
- Pilot (records never used): the mini run end to end (`state/run5mini/`) and the format pilot (`state/run5pilot/`).

**V3-058 (2026-10-30, after stage 2 and before any sealed score was viewed): read this for everything after stage 2.**
The rules are `docs/RUN5_STAGES_DESIGN.md` (locked with this file) and DECISIONS V3-058; they replace the "Final
design" and "Final check" items above.
- Stage 2 verdict and every confirmation: the gated baseline retention rule (`v3.analysis5.retention`): keep mains
  surviving Holm with a positive estimate; keep a pair only if it survives Holm and both its mains are kept; never +
  solely through a negative interaction; the design changes only if its direct paired contrast with the current best
  has its 95% t range above zero across base systems. Stage 2: D01 = FACTS.
- The sealed set opened on 2026-10-30 is retired: records unchanged, manifest with hashes in
  `state/run5/retired_V3_058.json`, never analysed (`Retired`).
- At most three stages (3, 4, 5). Each: Opus writes 4 candidates for each of 5 slots (layout, fact rules, build
  check, answering, worked example; one per archetype) from the firewalled evidence; mechanical checks (isolation,
  naming, syntax for the build check, lengths: rules half the current length to 2,500, build check and answering
  2,500, example 4,000; diversity; checker-clean examples; 3 attempts); human approval; screen of 21 designs x 320
  fresh items (advance: paired lead >= +0.015; at most 4 slots; futility stop); confirmation 2^m x 480 fresh items
  (Holm over mains + pairs; the retention rule; convergence stop under +0.005).
- Item sets frozen in `items/run5_stages.json` (in the lock): s3-s5 screen 80 and confirmation 120 bases each
  (validation), sealed2 180 bases (train); no overlap with anything used before.
- Final check after the last stage on sealed2: the final design vs A0P, vs ONT_LOOSE, vs A3, vs FACTS (if they
  differ), fixed sequence at alpha 0.05 (95% t range across base systems above zero); FACTS vs A0P and scores by k
  reported; the final design vs FACTS on the 500 hand-written items, paired across items (df 499).

## Run 4 (V3-037): read this first
Run 3 (2026-10-24 16:22 to 2026-10-25 09:14) met its stopping rule after 4 rounds without a new champion; finals A
and B ran, final C was stopped by the human at 46 of 600 episodes, and the run is archived in
`archive/run3_2026-10-25/` (its finals' results recorded but not analysed when run 4 was designed). **Run 4 is a new
experiment with its own rules, listed under "Run 4 rules" at the end; where they differ from the text in between,
the run 4 rules apply.** The run 1-3 sections are kept unchanged as the record of those runs.

## Run 3 (V3-034): read this first
Run 2 (2026-10-23 17:57 to 2026-10-24 08:48, paused by the human in generation 2) is archived in
`archive/run2_2026-10-24/` with its dev-data report; its sealed sets were never opened. **Run 3 is a new experiment
with its own rules, listed under "Run 3 rules" at the end; where they differ from the text in between, the run 3
rules apply.** The run 1 and run 2 sections are kept unchanged as the record of those runs.

## Run 2 (V3-030)
Run 1 (2026-10-21 to 2026-10-23) is archived in `archive/run1_2026-10-23/` with its dev-data report; the sealed sets
were not opened. **Run 2 starts from a fresh state with GPT-6 Luna as the only builder in evolution.** Where the text
below says "Claude builder" for selection or for H1, H3-H5, run 2 reads "GPT-6 Luna"; Claude takes part only in the
final stage, once per sealed item, as a transfer check. The run 2 rules are listed under "Run 2 rules" at the end.

## Question
Can an evolutionary process, with AI as the variation operator and pre-registered algorithms as selection, find a way
for an AI to construct and use a model of a described system that answers questions better than effort-matched
plain-language reasoning? Can it improve this generation on generation?

## Hypotheses (tested on the sealed final sets)
- **H1 (primary):** the best evolved design beats A0+ (effort-matched plain language, same builder) on end-to-end
  score, Claude builder, on the sealed v2 held-out set: a paired cluster-bootstrap 95% CI of the difference excludes 0.
- **H2:** the same holds for the small builder, GPT-6 Luna via Codex (does the design help AI generally, not only
  Claude?). Designs whose tools run model-written code (probabilistic, simulation) are Claude-only (V3-011).
- **H3 (process):** generations improve. The best full-evaluation e2e on dev rises from generation 0 to the last
  generation, with paired CI > 0 against the best generation-0 seed.
- **H4 (KF-1, from v2):** buildability limits e2e. Across fully evaluated designs, e2e increases with buildability.
  This is reported as a correlation with a CI, not as causal.
- **H5 (structured plain language, V3-006):** the best design with a structured plain-language core (sectioned or
  ontology) beats free plain language (A0+) on the sealed composite items, Claude builder: paired cluster-bootstrap
  95% CI of the difference excludes 0. Secondary: the difference grows with complexity, i.e. the slope of the
  per-item difference against log2(k subsystems) has a bootstrap 95% CI above 0. Reported for GPT-6 Luna too.
- **H6 (structure vs model capability, V3-009, human's hypothesis):** structure helps the smaller builder more than
  the frontier builder: the H5 difference (best structured design minus free plain language) on sealed composite items
  is larger for GPT-6 Luna than for Claude (paired bootstrap 95% CI of the difference of differences above 0). Secondary,
  descriptive: whether evolution's best designs for each builder differ in strictness (strict vs loose ontology,
  tags, granularity, header count).

## Fixed design
- **Builders:** Claude Sonnet 5.5 (headless, arm tools only, Claude subscription) and GPT-6 Luna (headless Codex,
  allowlisted arm tools only, ChatGPT Pro subscription; V3-013 replaces Qwen3 235B), with identical
  prompts.
- **Controls:** A0P (v2 A0+, effort-matched plain language) on both builders.
- **Items (V3-006 ladder, script-graded composite items at k = 1, 2, 4, 8 subsystems):**
  - quick build test: 5 composite items (1/1/2/1);
  - screening: 30 composite items (6/8/8/8);
  - full evaluation: 10 v2 dev items (AI-graded) + 48 composite items (12 per k);
  - micro pool: 20 composite items (5 per k), disjoint from the sets above.
  The fixed ids are in the lock.
- **Final sets (sealed until the final hash lock):** the v2 held-out set (never opened in v2), the v2 generalisation
  set, and generated test-split items (including the families held out of evolution).
- **Grading:** generated items are graded by script; v2 items by the AI grader chosen in the grader test (agreement ≥
  0.85 with a Claude Opus referee, V3-004). A frontier model audits 5% of final answers, and final
  answers are double-graded.
- **Selection:**
  - successive halving (quick build test → 10 items, top half → 30 items, top 3 to full);
  - NSGA-II on e2e, buildability and −output tokens (Claude builder, full set);
  - MAP-Elites niches = core × build protocol;
  - binary tournaments (each design counted once; ties on rank and crowding go to Claude e2e + Luna e2e,
    V3-026); 3 islands, with 2 migrants every 3 generations.
- **Micro level (V3-015, V3-017):** before breeding, 6 code-only tweaks (param, simplify, numeric) of each of the top
  3 designs run on the small builder on 5 items drawn for that generation from a separate 20-item micro pool, paired
  against their parent. Candidates (build on >= 3 items, beat the parent) must beat the parent again on 5 fresh pool
  items; the first confirmed candidate per design joins the generation's children. No operator credit for micro.
- **Variation:** 20 children per generation from the operators and starting shares in config/v3.json. Shares are
  adapted by win rates, with a floor of 0.03; a child wins if it beats its first parent in screening or reaches
  full evaluation (V3-027). Every child declares 1-3 parents.
- **Screening (V3-023, from generation 2):** two tracks. Every child that passes the quick test is screened on Claude
  and (if it can build the design) on the small builder; each track halves on its own builder's score (better half
  after 10 items, ranked on 30). Full evaluation takes Claude's top 2 and the small builder's top 1 not already picked.
  Generations 0 and 1 screened on Claude only; generation 1 gets its small-builder track once before generation 2
  breeds (catch-up, same rule). The NSGA-II objectives add the small builder's full e2e. Micro bases: the top 2 by
  Claude plus the best by the small builder.
- **Improvement rule:** non-dominated, and a paired CI excluding 0 on e2e or buildability against each declared parent
  that has a full evaluation, and against the best control; from V3-023 judged per builder, passing on either counts.
- **Stopping:** 4 generations without improvement, 20 generations, budget exhaustion, or the human's PAUSE.
- **Final selection:** up to 3 finalists from the NSGA-II front, in distinct niches, chosen by the highest Claude e2e
  within each niche. Ties are broken by buildability. From V3-023 the front's best design on the small builder is
  always a finalist (replacing the last pick other than the structured one kept for H5).
- **Effort:** output tokens and turns are recorded for every run. Designs outside ±30% of A0+ effort are flagged in the
  report.

## Run 2 rules (V3-030, locked before run 2's generation 0)
- **Builder:** GPT-6 Luna only (`evolution.primary_builder = "codex"`): quick test, micro, screening (one track) and
  full evaluation. Designs Luna cannot build (probabilistic, simulation cores) fail the quick test.
- **H1 (primary, run 2):** the best evolved design beats A0+ on end-to-end score, **GPT-6 Luna builder**, on the sealed
  v2 held-out set: paired cluster-bootstrap 95% CI of the difference excludes 0.
- **Which design H1 tests (V3-032):** "the best evolved design" is the one finalist with the highest dev full-set Luna
  e2e, named in the final lock before the sealed sets are opened. H1 passes or fails on that design alone. The other
  finalists are secondary results: their p-values against A0+ are Holm-adjusted across the secondary finalists and
  none of them can make H1 pass. H2 uses the same primary design. Dev-set comparisons of the best design with the
  control (the dashboard panel) are selection-biased and are descriptive only, never evidence for H1.
- **H2 (run 2):** the same holds for Claude, which never took part in evolution (transfer).
- **H7 (complexity, V3-033, human's hypothesis, secondary):** the H1 primary design's lead over A0+ is larger on the
  most complex systems than on the simplest. On the sealed generated composite items, GPT-6 Luna builder: (mean
  difference primary minus A0+ at k = 8) minus (the same at k = 1), paired cluster-bootstrap 95% CI above 0.
  Equivalently, the primary design loses less from k = 1 to k = 8 than A0+ does. Reported for Claude too, and with the
  per-k differences. Computed in the report from the per-item records; it cannot make H1 pass or fail. Proposed after
  a dev-data observation (run 2, generation 2: Velvet Owl +0.12 [+0.06, +0.18] over A0+ at k = 8, +0.07 at k = 1),
  so it counts only if confirmed on the sealed items.
- **H3, H4, H5 (run 2):** as above, on Luna's results (H3: best Luna full e2e, last generation vs the best seed; H5 on
  Luna, reported for Claude too). **H6** unchanged (Luna's H5 difference minus Claude's).
- **Variation:** 100 children per generation.
- **Screening:** successive halving on Luna (better half after 10 items, ranked on 30); the top 6 go to full
  evaluation.
- **Full evaluation:** 120 items (10 v2 dev + 110 composite items, 27/27/28/28 at k = 1/2/4/8).
- **Selection:** NSGA-II on Luna's e2e, buildability and −output tokens; tournament ties on Luna e2e; micro bases are
  the top 3 by Luna e2e; operator credit and failure targets from Luna's results.
- **Improvement rule:** non-dominated, and a paired CI excluding 0 on Luna e2e or buildability against each declared
  parent and the best control.
- **Stopping:** unchanged (4 generations without improvement, 20 generations, budget, or the human's PAUSE). Luna's
  weekly-allowance guard pauses the run at 90%; it resumes after the weekly reset.
- **Final:** up to 3 finalists from the front by Luna e2e (distinct niches, one structured design kept for H5); on the
  sealed sets Luna runs each item 3 times and Claude once.
- **No change after launch** except pause and resume; a true bug fix is recorded as a new decision with its voided runs.

## Run 3 rules (V3-034, locked before run 3's generation 0)
- **Goal (human's words, 2026-10-24):** "a creature that describes complex systems with the highest accuracy for
  interactions with AI and building systems". Evolution optimises one number: GPT-6 Luna's end-to-end score (e2e) on
  the full set. The human's principle: "optimize breeding for best outcomes, not mimic actual evolution". Run 3 is a
  directed search (code: `v3/directed.py`), not a simulation of natural evolution.
- **Builder:** GPT-6 Luna only, in evolution and in finals A and B. Claude takes no part as a builder or test subject
  (no H2, H6); Claude Opus remains the designer for AI operators. A separate evolution for frontier models may follow.
- **Seeds (from scratch, reviewed one by one with the human after a literature search, `docs/seeds_run3.json`):**
  A0P (control), SEC_A1, ONT_LOOSE, SEC_FLOW, G1C3, A5 (rules), and three new sectioned seeds: FACTS (key-value fact
  lines), STATES (state machine, WHEN/IF/THEN transitions), SPEC (EARS-style build specification). ONT_BASIC is
  dropped. Nothing is carried over from runs 1-2.
- **Items:** fresh bases (validation split from 1,200,000; runs 1-2 used 1,500,000-1,616,500): quick 5, screening 30,
  full 120 (10 v2 dev + 110 composite, 27/27/28/28 at k = 1/2/4/8), three extension sets of 120 (30 bases x k = 1, 2,
  4, 8, from 1,650,000; the third added by V3-036). Fresh confirmation sets: 120 items each (30 bases x 4 k) from the train split, each starting
  after the last base of the previous one, never reused. The fixed ids are in the lock; confirmation sets are recorded
  in the state when drawn.
- **Each round:** 36 children (`evolution.children_per_generation`), in a fixed mix: 50% **fix** (AI designer, operators
  point / repair / guidance / big_jump, shown the goal and parent 1's own failure profile: construction vs reasoning
  counts, score and build rate by k, and its most frequent checker problems with quoted item terms blanked), 30%
  **combine** (transplant: one leader plus the change that made another leader better than its own parent, the
  second leader from another starting line when one exists; or a gene-level swap), 20% **explore** (simplify, param,
  numeric; graft only once the gene library has entries). Within each kind, operator shares adapt to win rates (floor
  0.03); every failed attempt to make a child counts as a try. About 10% of children (4 of 36) are **reserve**
  children: their parent is the best design of a family (core kind) other than the champion's. Parents are the
  **leaders**: the top 6 fully evaluated designs by Luna full e2e, picked with linear rank weights. One pool, no
  islands, no migration, no micro level.
- **Screening:** quick build test (5 items, >= 3 built), successive halving on Luna (better half after 10 items,
  ranked on 30), the top 4 go to full evaluation. A child wins for operator credit if its 10-item screening score beats
  its first parent's on the same items, or it reaches full evaluation.
- **Champion:** at the start, the seed with the highest Luna full e2e. The champion is the design run 3 currently
  backs; it changes only by the two-step rule below.
- **Step 1, growing proof (as amended by V3-036 from round 1's proof step on):** the round's best fully evaluated
  child, whatever its full e2e, and any other fully evaluated child whose full e2e exceeds the champion's, are compared
  with the champion and with the control (A0P), paired, Luna e2e, cluster bootstrap (clusters: composite items by base
  system), 2,000 resamples, **on fresh items only**: the extension sets, which no design was selected on. Step one is
  ext1 (120 items); if not proven and its lead over the champion is still positive, ext2 and then ext3 are added and
  it is tested again on the pooled extension items: 120 -> 240 -> 360. Proven = the 95% CI is above 0 against both. A
  lead of zero or less, or no proof at 360, ends it. The full set is not used to judge, because the children and the
  champion were all selected on it. (Up to three looks at the data; step 2 guards against the extra chances.)
- **Step 2, fresh confirmation:** a proven candidate is tested once on the next fresh 120-item confirmation set, with
  the champion and the control. It becomes the new champion if its mean is above both of theirs on that set. At most
  one confirmation per round; other proven candidates queue and are proven again against the champion of the day.
  The new champion's genes become the graft library.
- **Stopping:** 4 rounds without a new champion, or 20 rounds, or the human's PAUSE, or an allowance guard. Budget:
  Luna's weekly-allowance guard pauses at 90%; OpenRouter spend (grader and final C) is capped by the existing cash
  limit in `config/v3.json` (budget).
- **Built-in finals (`v3/final3.py`), run automatically when evolution stops; finalist = the champion only, named in
  `state/FINAL_LOCK.json` before any sealed item is opened; control A0P:**
  - **Final A, accuracy (sealed sets, Luna, 3 repeats per item):** the v2 held-out set, 60 generated test-split items,
    and the composite test ladder (15 bases x k = 1, 2, 4, 8, from 2,500,000).
    - **H1 (primary):** the champion beats A0P on Luna e2e, pooled over the three sealed sets (mean over repeats per
      item): paired cluster-bootstrap 95% CI of the difference above 0. Per-set results are reported.
    - **H7 (secondary, V3-033):** on the composite test items, (difference at k = 8) minus (difference at k = 1) has a
      CI above 0 (clusters: base systems). Per-k differences are reported.
    - **H5 (secondary):** if the champion has a structured plain-language core, its difference on the composite test
      items and the slope against log2(k); not tested otherwise.
    - **H3 (descriptive):** the champion vs the best seed on the full dev set, and the chain of champions with their
      confirmation results (the fresh confirmations are the unbiased estimates of each step's gain).
    - **H4 (descriptive):** Spearman correlation of e2e with buildability across fully evaluated designs.
  - **Final B, interaction and building (Luna):** 20 systems from the test split (bases from 2,700,000), each built up
    over four turns. Turn 1: the builder represents a system with one subsystem and answers its questions. Turns 2-4:
    the builder gets only its own representation from the previous turn and the text of the newly added subsystems
    (1, then 2, then 4 more: k = 2, 4, 8), the earlier descriptions are gone; it rebuilds the complete representation,
    then a fresh answer phase answers every question so far from that representation alone. Champion and control.
    - **H8 (key secondary):** the champion's mean score over all turns beats the control's (CI above 0, clusters:
      systems).
    - **H8 compounding (secondary):** (difference at turn 4) minus (difference at turn 1) has a CI above 0, i.e. the
      champion's lead grows as the system is built up.
    - Retention (descriptive): of the questions answered correctly at one turn, the share still answered correctly at
      the next.
  - **Final C, transfer to other models (once per item; V3-035):** the composite test ladder (60 items), champion and
    control, on five pre-named models: four on OpenRouter, Qwen 3.8 Flash (`qwen/qwen3.8-flash`, reasoning off), Gemma 4
    26B (`google/gemma-4-26b-a4b-it`, no reasoning setting), Mistral Small 2603 (`mistralai/mistral-small-2603`, none),
    gpt-oss-20b (`openai/gpt-oss-20b`, reasoning effort low); and Claude Haiku on the human's Claude subscription
    (`cl_haiku`, Claude Code headless, the run 1 Claude builder with model `haiku`, under the Claude hourly cap and
    weekly guard). Evolution stays Claude-free; Haiku is only a transfer model. No spare.
    - Answers in final C are read leniently (the evolution and finals A and B keep the strict `ANSWERS:` reader): also
      `FINAL ANSWER {...}` (the A0P prompt asks for a FINAL ANSWER line) and, failing any JSON object, explicit
      `s<j>.p<n>: <option>` pairs after the last ANSWERS / FINAL ANSWER marker. Same reader for champion and control.
    - A model with paired champion/control records on fewer than half of the 60 items (rate limits, empty responses,
      any failure after the usual transient retries) is reported as unusable and left out; the human accepted this in
      advance ("if other models cannot be used like mistral for whatever reason that is fine"). H9 needs at least 2
      usable models, otherwise it is reported as not run.
    - **H9 (key secondary):** pooled over the usable models (item x model pairs, clusters: base systems) the
      champion's difference has a CI above 0, and the champion is ahead on more than half of the usable models.
    - Harness note for the paper: Haiku works through Claude Code's tool harness, the OpenRouter models through the
      v2 episode loop; each model's champion/control comparison uses one harness.
    - History: V3-034 named four OpenRouter models with Nemotron as spare; the pre-launch check failed most of them
      (unbounded reasoning, answer format), so final C waited for V3-035 (`final.transfer_ready`), fixed on
      validation items with seed designs only while round 0 ran, before any final C record.
  - Every final is piloted on non-sealed validation items before the lock (`python -m v3.final3 pilot`).
  - After the finals: `runs/final/analysis3.json` and a paper kit (`runs/final/paper_kit/`: the champion's genome, the
    exact system prompt and build instruction, an example representation it wrote on a dev item), for the human's
    white paper "Describing Reality for AI".
- **No change after launch** except pause and resume; a true bug fix is recorded as a new decision with its voided runs.

## Run 4 rules (V3-037 and V3-038, locked before run 4's generation 0)
Same goal, builder, designer, control (A0P) and champion rule as run 3 (growing proof on fresh items, then a fresh
confirmation). Code: `v3/directed4.py` (class Directed4); run 3's `v3/directed.py` is unchanged. Each change below was
reviewed one by one and approved by the human on 2026-10-25, after run 3 and an offline replay of its screening
(`paper/side/replay_screening.txt`: run 3's 10-item screen held only k = 1-2 items; re-running the same child on the
same 10 items moved its mean by 0.084 on average, more than the 0.071 spread between children).
- **Starting designs (round 0):** run 3's nine seeds (A0P control, SEC_A1, ONT_LOOSE, SEC_FLOW, G1C3, A5, FACTS,
  STATES, SPEC) plus four of run 3's near-winners carried over with unchanged genes (`docs/seeds_run4_carryover.json`):
  Stormy Marten, Lanky Finch, Little Marten (FACTS family) and Lucky Beetle (STATES family). Each keeps its run 3
  family for the family rules. The champion at the start is the starting design with the highest Luna full e2e on run
  4's own full set (it need not be FACTS).
  **V3-039 (2026-10-25, after round 0 and round 1's breeding, before any round 1 child was scored beyond its quick
  test): the four carried-over designs are dropped from the run.** They were near-copies of FACTS (Stormy Marten
  differs from FACTS only in the numeric gene; Little Marten and Lucky Beetle only in the core), so five of the six
  leaders were one design: smart combining failed all 10 attempts, no three-parent trio existed and 3 of 20 children
  could not be made. The run restarts round 1 from the nine seeds only: round 0 is recomputed from the stored records
  of the same 120 items (no new runs), so the starting champion is the best of the nine. The carried-over designs'
  round 0 scores stay on record and are reported; round 1's first breeding is archived unscored.
- **Items (all fresh; `scripts/check_item_ranges.py` checks that no base system of any run 4 set overlaps any item
  used before or another run 4 set):** one ordered main set of 120 items, 110 composite (validation split from
  1,300,000; 27/27/28/28 at k = 1/2/4/8) followed by the 10 v2 dev items. The order is interleaved so that every stage
  is spread over the four system sizes: items 1-5 = quick, 6-35 and 36-65 = the two screens, 66-120 = the rest of the
  full set. Extension sets ext1-ext3 (120 each, 30 bases x 4 k) from the train split at 100,000; fresh confirmation
  sets from the train split at 500,000; 8 **practice** items (4 bases x k = 4, 8, train split at 300,000), which are
  never used to score or select designs.
- **Each round, children (improvements 3 to 5):** 20 children: about 30% **fix** (point, guidance; repair only when
  the parent's build rate is below 0.95, otherwise it becomes point), 35% **combine** (smart_combine 0.6, transplant
  0.25, swap 0.15), 35% **big ideas** (big_jump 0.4, answer_method 0.4: a change to how Luna uses its representation
  to answer, limited to the answering and query layer genes (V3-038); simplify, param, numeric, graft 0.05
  each). Shares within each kind adapt to win rates (floor 0.03), failed attempts count as tries. About 1 in 5
  children are **reserve** children: their parent is the best design of one of the three strongest other families
  (starting design they descend from) than the champion's.
- **Designer report card (improvement 2):** AI fix and big-idea operators get parent 1's report card: overall score,
  build rate and score by system size on its full evaluation; build problems only if builds still fail; and, from the
  practice items (run once per parent and kept), its accuracy by size, up to 8 questions it answered wrongly (the
  question, its answer, the correct answer) and its own representation of one k = 8 practice system. The goal text
  says builds already pass and the score is lost on large systems. The designer never sees scoring, extension,
  confirmation or sealed items or their answers.
- **Nine genes (V3-038):** the guidance gene is split in three: guidance (how the builder writes its
  representation), answering (how it answers questions from it; shown to the builder under "When you answer questions
  from the representation", nothing when empty) and examples. Designs written before the split (all starting designs
  and carry-overs) get an empty answering gene and their examples moved to the examples gene; their builder prompt and
  their id are unchanged (the id is the hash of the pre-split form). The guidance operator may change guidance and/or
  examples; answer_method only answering and the query layer. The creatures draw answering as ears, crest and brows
  and examples as beads and cheeks.
- **Three-parent combining (V3-038):** 2 combine slots per round that are not shelf slots (`three_parent`) are
  smart_combine3: three parents, the best fully evaluated design of three of the (up to) four strongest families
  among the leaders, the shelf and the champion; the child takes each gene whole from one of them, with a reason per
  gene, and every parent must give at least one gene that matters (not the schema gene of a plain-language core) and
  that only it has. The designer may answer that no such mix fits; after two tries the slot becomes an ordinary
  combine. Bad mixes are judged against all parents. Both smart combines also require each parent to give a gene that
  matters.
- **Champion children (V3-040, from round 2):** in every round from round 2 on, the first fix slot and the first
  big-idea slot that are not reserve slots breed from the champion: a fix child from its report card (the fix operator
  drawn as usual) and an answer_method child. Each keeps the champion as parent for two tries, then breeds as usual.
  The other 18 slots are unchanged.
- **Smart combining and the shelf (improvement 4):** smart_combine shows the designer both parents' strengths cards
  (score by size, build rate, practice accuracy by size) and per-gene evidence (mean full e2e of fully evaluated
  designs carrying each parent's variant minus the mean of all), and recent bad mixes; the child takes each of the 9
  genes whole from one parent, at least one from each, with a reason per gene; incompatible results are refused. A
  smart-combine child whose mean on its screened items is below both parents' on the same items is recorded as a bad
  mix and shown in later prompts. The **shelf**: a child whose growing proof ends within 0.02 of the champion (last
  step's estimate) joins it; half of each round's combine slots pair two shelf members or a shelf member with the
  champion (smart_combine).
- **Screening (improvement 1, as adjusted by the human after the replay):** scores add up across stages (one record
  per item, reused). Quick: items 1-5, pass if >= 3 builds, no ranking. Screen 1: items 1-35 for every child that
  passed; a child is cut only if it is clearly behind the champion: the paired cluster-bootstrap 95% CI (2,000
  resamples, clusters = base systems) of its difference to the champion on the same items lies entirely below 0.
  Screen 2: items 1-65 for the rest; clear losers cut again; the top 6 (`full_top_k`) by the paired estimate go on,
  one place held for the best child of a family other than the champion's if none of the top 6 is from one and it is
  not a clear loser. Full: topped up to all 120 items. A child wins for operator credit if its 35-item score beats its
  first parent's on the same items, or it reaches full evaluation.
- **Families (improvement 5):** a family share over 70% of a round's children for two rounds in a row is logged as an
  alarm (and shown on the dashboard); the reserve and the family place above keep other families in play.
- **Growing proof and fresh confirmation:** as run 3 (V3-036), on run 4's own ext1-ext3 and confirmation sets.
- **Stopping (improvement 6):** 8 rounds without a new champion, or 20 rounds, or the human's PAUSE, or an allowance
  guard.
- **Finals:** as run 3's built-in finals A, B, C (H1, H3-H5, H7, H8, H9 unchanged), with fresh sealed sets: final A on
  60 generated test-split items from seed 2,100,000 and the composite test ladder of 15 bases x 4 k from 2,600,000
  (H1 pooled over these two sets); final B on 20 systems from 2,800,000; final C on that composite ladder. **Run 3's
  sealed sets are retired** (generated 2,000,000+, ladder 2,500,000+, interaction 2,700,000+) **and so is the v2
  held-out set**, which run 3's final A opened. The FINAL_LOCK records run 4.
- **Pilot:** `python -m v3.run pilot` with run 4's code on a small slice (validation 1,450,000: stages of 5/9/13
  items, 8 children, 2 to full, six starting designs) before the lock is final; its state and records are separate.
- **No change after launch** except pause and resume; a true bug fix is recorded as a new decision with its voided runs.

## Not decided by the AI
Survival, finalists, improvement claims and stopping are computed by code from the rules above. AI agents only propose
gene changes (operators), build models (builders) and grade prose answers (grader).

## Deviations from the spec already recorded
V3-001: the micro level was dropped, then reinstated in a different form (V3-015, V3-017: tweaks judged on the small builder, with confirmation). V3-002: E and F are reported, not
selection objectives. V3-004: a Claude Opus referee, not a human spot check, is the grader-test reference.
V3-023: two screening tracks and a small-builder objective (above). V3-026: one island entry per design and tournament ties broken by score (above). V3-027: reaching full evaluation counts as an operator win (above). V3-006: after the pilot, structured plain-language cores, a complexity ladder of composite items and H5 were added;
the item counts above are replaced by the calibrated ladder recorded in the lock. V3-010 to V3-013: the small builder is GPT-6 Luna through Codex; Qwen3 235B is retired. V3-014: the Claude
throttle is 120 runs per hour. V3-029: Claude's cap is 360 calls per hour, 12 at a time, with a weekly-allowance guard that
pauses the run at 85% (as Luna's pauses at 70%).
V3-030: run 2 is a clean run with GPT-6 Luna as the only builder in evolution, 100 children, 6 fully evaluated, 120
full-evaluation items, Luna up to 10,000 calls per hour (12 at a time) with its weekly guard at 90%; decided after
seeing run 1's dev results (see "Run 2 rules").
V3-031: transient episode failures are retried up to 2 times (failed attempts kept as record.void files); run 2 restarted from zero with this rule, the aborted start archived.
V3-032: H1 tests one pre-named design (the finalist with the highest dev Luna e2e); other finalists are secondary, Holm-adjusted; decided during run 2 generation 2, before any sealed data was seen.
V3-033: H7 added (secondary): the primary design's lead over A0+ is larger at k = 8 than at k = 1 on sealed composite items; proposed from a run 2 dev observation, before any sealed data was seen.
V3-034: run 3, a new experiment: directed breeding (fix / combine / explore mix, leaders as parents, reserve, one
pool, no micro), a champion changed only by a growing proof plus a fresh confirmation, new seeds from a literature
search, and built-in finals A (accuracy, H1 pooled over the sealed sets), B (interaction, H8) and C (transfer to four
OpenRouter small models, H9); no Claude builder. Decided after seeing run 2's dev results and the cross-breeding
probe; run 2's sealed sets were never opened (see "Run 3 rules").
V3-036: growing proof judged on fresh items only (three extension sets, 120 -> 240 -> 360), and the round's best
fully evaluated child enters it even below the champion's full score; decided in round 1 before its proof step,
because the starting champion's full-set score carries selection luck (best of nine on those items).
V3-035: final C settings fixed after launch, before any final C data: per-model reasoning settings, lenient answer
reading for final C only, Claude Haiku (subscription) added as a fifth model, no spare, unusable models left out of H9
(min 2 usable). Decided during run 3 round 0 on validation items with seed designs only.
V3-037: run 4, a new experiment: screening on one interleaved main set whose scores add up, cutting only clear losers
against the champion (35 then 65 items, top 6 to full); 20 children with a fix / combine / big-idea mix; a designer
report card with practice items; smart combining and a near-winners shelf; family places; 8 rounds without a new
champion to stop; run 3's near-winners as extra starting designs; all-new item sets and sealed sets (run 3's and the
v2 held-out set retired). Decided after run 3 ended and its screening was replayed offline; run 3's finals' results
were not analysed (see "Run 4 rules").
V3-038: before run 4's pilot, the guidance gene is split into guidance, answering and examples (9 genes; old designs'
ids and prompts unchanged) and 2 combine slots per round take genes from three parents of three families. Decided by
the human before any run 4 data (see "Run 4 rules").
V3-039: after run 4's round 0, the four designs carried over from run 3 are dropped (they were near-copies of FACTS
and left breeding without diversity); round 1 restarts from the nine seeds, round 0 recomputed from stored records,
round 1's first breeding archived unscored. Decided by the human after seeing round 1's parents and children, before
any round 1 screening result (see "Run 4 rules").
V3-040: from round 2, every round breeds one fix child and one answer_method child from the champion. Decided by the
human during round 1's first screen, after the champion was drawn as the main parent of none of round 1's 13
single-parent children; applies from round 2, so round 1 is unaffected (see "Run 4 rules").
V3-055: the run 5 state save retries a briefly locked state file for up to about 10 s (Windows file lock at the end
of stage 1 stopped the loop; no rule, item or score changed).
V3-056: the stage 2 "+" texts are written with a summary of the winner's stage 1 evidence on the generated items
(firewalled: no hand-written result, family score, item, question or answer key), not from first principles alone;
the texts written blind are archived (`state/run5/plus_blind_1.json`), unapproved and unused. Decided by the human
after stage 1 ended and before any stage 2 run; stage 2 still reuses the stage 1 generated items, so the stage 2
effects may be slightly optimistic, and the sealed final check is the unbiased estimate.
V3-057: the evidence brief states how the build tools work, read from the design (each save replaces the whole
notes; after each save the builder sees the first checker problem and how many more remain), correcting a false
V3-056 sentence; texts written from an earlier brief are archived (`plus_superseded_<n>.json`), unapproved and unused.
V3-058: after stage 2 (complete) the locked final-design rule picked D18, which scored lower than FACTS; the final
check was stopped by the human 67 minutes after the sealed set was opened, before any sealed score was viewed. The
rule was replaced by the gated retention rule (stage 2 verdict: FACTS), the opened sealed set retired unanalysed,
stages 3-5 added on fresh items, and the final check moved to a fresh sealed set after the last stage (see above).
