# Run 5 build plan (DRAFT for the human's approval, 2026-10-28)

Status: draft. Nothing is built. The design it builds is `docs/RUN5_DESIGN.md` (all six open questions decided
2026-10-28). This plan says what gets built, in what order, how each piece is checked, and where the human decides.
Part C is the "How it works" text for the dashboard: the whole process explained in plain words beside the technical
detail.

---

## Part A. What gets built, in order

Each step lists what is made, how it is checked, and whether the human must approve before the next step. Steps 1 to
4 do not depend on each other and can be built side by side; step 5 onwards need them.

### Step 0. Record the decisions (human gate)
- `DECISIONS.md`: V3-042 "Run 5: a designed experiment in two stages" (the design, the six decisions, why) and
  V3-043 "Own code allowed for Luna, in Docker only" (lifts `REFUSED_CORES` for A3 and G1C1; a safety-rule change).
- (Done 2026-10-28: `RUN5_DESIGN.md` section 2 updated to fifteen founders.)
- **Human:** approve this plan; that approval is recorded in V3-042.

### Step 1. Docker sandbox for model-written code
Made:
- `sandbox/Dockerfile`: a small Python image (pinned version, numpy and scipy only, no compilers, no network tools),
  a non-root user, the v2 audit-hook runner (`rpp-core/harness/sandbox_runner.py`) copied in as a second layer.
- `v3/sandbox.py`: `run_code(code, inputs) -> result` starts one throwaway container per call:
  `--network none --read-only --tmpfs /scratch:size=64m --memory 512m --cpus 1 --pids-limit 64 --user 1000
  --security-opt no-new-privileges --cap-drop ALL --rm`, a 30-second wall-clock limit (the container is killed after
  it), output capped at 64 KB.
- The A3 and G1C1 cores' code-running tool calls `run_code` instead of running Python on the host.

Checked (`tests/test_sandbox.py`, all must fail to do harm): open a network connection; read a planted file outside
/scratch; write to the host disk; start a shell or another program; fork without limit; allocate 10 GB; loop
forever; fill the disk; read environment variables holding keys (none are passed in). Plus: 20 containers at once
(the run's concurrency) without errors, and the time each call adds.

### Step 2. The three new founders and their cores
Made:
- **JSON core** (`v3/cores/jsoncore.py`): the notes must parse as one JSON object; if not, Luna gets the parse error
  and one repair turn. Tools: `lookup(path)` (a key path such as `parts.pump.flow_rate`) and `search(text)`.
- **MARKDOWN core** (`v3/cores/markdowncore.py`): headings, tables with consistent columns, rule bullets; one repair
  turn on a broken table. Tools: `lookup(heading or row name)` and `search(text)`.
- **OWN**: the plain notes core with a neutral instruction and no format check; `v3/formclass.py` records by code
  which form Luna chose (valid JSON, Markdown tables or headings, bullets, prose, code-like).
- Three design briefs (JSON, MARKDOWN, OWN) written by Claude Opus from first principles, at the same level of effort
  as the existing founders.
- `v3/genome/seeds/JSON.json`, `MARKDOWN.json`, `OWN.json`.
- Codex allowlist: the new tool names (`lookup`, `search` where new, `run_code`) added to `ALLOW` (widening it needs
  this approval, recorded in V3-042).

Checked: unit tests per core (valid, broken and repaired notes; lookups); all 15 founders validate and render;
`usable_for` accepts all 15 for Luna.
**Human:** read the three briefs before they are frozen.

### Step 3. Items
Made:
- **Generated:** the main set of 360 (90 bases x k = 1, 2, 4, 8) and the sealed final set of 720, from seed ranges no
  earlier run used (`scripts/check_item_ranges.py` proves no overlap). The 720 are written to a sealed file that the
  run code refuses to open before the final check.
- **Hand-written style, 500 items** (`items/hand5/`), by a script that calls Claude Sonnet on the Claude plan:
  1. A plan of about 170 systems balanced over domains, kinds (continuous, discrete, stochastic) and messiness
     (realistic, adversarial), each with three question types; written by code, so the balance is exact.
  2. Writer agents (Sonnet), one system each: design the system first (its true parts, values, causes), then write
     the messy document about it (logs, emails, meeting notes, about 6,000 characters, with noise and
     contradictions as the messiness level says), then the three questions and answer keys (summary, key points,
     typical wrong answers), following the v2 construction protocol.
  3. Checker agents (Sonnet, a fresh agent that did not write it), given the document: answer the question; the
     answer must reach the key's key points, and the key must agree with itself and with the document.
  4. Leak check (DeepSeek v4.1 Flash), given the question without the document: it must fail. An item it answers is
     a give-away or something the models already know, and is replaced.
  5. Any failing item is rewritten once, then dropped and a new one written, until 500 pass.
  6. Grader check: on 50 items, Luna-style answers (from a pilot run) are graded by DeepSeek and by a Claude Opus
     referee; agreement must be at least 0.85 (the threshold run 4 used).
- None of these agents sees a design; no design agent sees these items. The 27 v2 dev items and the v2 held-out
  split are not used or opened.

Checked: counts and balance by code; every item passes steps 3 and 4; grader agreement.
**Human:** spot-check about 20 hand-written items (a page shows document, question and key side by side).

### Step 4. The run engine: `v3/run5.py`
One simple engine replaces run 4's breeding loop (archived code stays in place, unused).
- **Jobs:** a stage is a list of (design, item) pairs; every pair runs exactly once. Records are written per run
  (as now), so a restart resumes where it stopped and never repeats a finished pair.
- **Each run:** the existing two-phase episode (build the notes without the questions, then answer from the notes
  alone), Luna through Codex, as in run 4.
- **20 at a time:** a pool of 20 workers; the weekly allowance guard pauses at 90% and resumes after the reset;
  `state/PAUSE` stops new runs and lets running ones finish; started only by `scripts/run_v3_loop.cmd`.
- **Failures:** a run that fails for a passing reason (connection, timeout) is retried twice, then left out and
  reported; never scored 0. A tool call outside the allowlist voids the run.
- **Stages in order:** `pilot` -> `stage1` -> `stage2` -> `final`. Stage 2 will not start until the five "+" texts
  are frozen and the human has seen them; the final check opens the sealed 720 only once, after stage 2.
- **Stage 2 designs:** built by code from the winner plus the frozen "+" texts, all 32 in standard order; the fit
  check (every combination renders; no answering text names a part a layout lacks) must pass.
- **Order of jobs:** items in a fixed shuffled order, designs interleaved, so if the run is paused early every design
  has about the same number of items done (a fair partial result).

Checked: `tests/test_run5.py` with a fake builder: every pair once, resume after a kill, retries, pause, sealed set
refused, stage order enforced, 32-design table and fit check.

### Step 5. The analysis: `v3/analysis5.py`
Code only, no AI. Implements `RUN5_DESIGN.md` section 6 exactly:
- Stage 1: paired means and 95% ranges (bootstrap over items, 10,000 resamples, fixed seed); the "could be best" set
  (Hsu); leads over the control with Holm; by group, by k, by family; OWN's chosen forms.
- Hand-written: the same for the 500, plus the rank agreement (Spearman) with a range.
- Stage 2: the 32 design means; five effects and ten pairs; the noise estimate from the 16 higher interactions; t
  tests (16 degrees of freedom), Holm over the seven pre-registered tests; the half-normal (Lenth) check; the final
  design (highest fitted score).
- Final check: paired leads of the final design over the stage 1 winner and the runner-up, on the 720 and on the 500.
- Writes `runs/run5/analysis.json` (read by the dashboard) and a report table for the paper.

Checked, before any real run: simulated data with known answers (founders with set gaps; a factorial with set
effects and noise sigma = 0.18, tau = 0.01). The code must recover the effects within their ranges, the Holm
false-alarm rate must be about 5% over 1,000 null simulations, and the power to find a 0.014 effect must be about 80%
(the figure in the design). This proves the maths before Luna spends a run.

### Step 6. The dashboard (streamlined)
See Part B. Built against a sample status file first, then the live feed. `scripts/check.py` renders it headless and
fails on any JavaScript error.

### Step 7. Pilot, part 1: the mini run end to end (about 100 runs; human, 2026-10-28)
"your pilot needs to include a mini run with say 100 test of the entire process to check everything works end to
end". The real engine, the real analysis and the real dashboard, on pilot items only (a separate seed range and six
extra hand-written-style items that are not among the 500), with every stage and every handover:
| Mini stage | What runs | Runs |
| --- | --- | --- |
| Stage 1 | 4 founders (A0P, FACTS, JSON, A3) x 6 generated + 2 hand-written items | 32 |
| Winner | chosen by the stage 1 rule in code; runner-up recorded | - |
| "+" texts | Opus writes the five "+" versions for the mini winner; fit check; frozen; shown on the dashboard | - |
| Stage 2 | all 32 designs built by code; the 31 new ones x 2 items (design 1 reused) | 62 |
| Final check | a mini sealed set (4 generated items), opened through the seal; 4 designs; the final design on 2 hand-written | 18 |
| Analysis | every output of `analysis5.py` (numbers meaningless at this size; it must run and fill every field) | - |
| Total | | about 112 |
Checks: every handover happens without a hand touching anything; a pause and a resume in the middle of stage 2
lose nothing; a forced failure is retried and reported; the sealed set refuses to open early; the dashboard shows
every tab filled; the analysis matches a hand calculation on these few numbers. Then the mini records are archived as
pilot data.

### Step 7b. Pilot, part 2: the format pilot (480 runs, about 1.5 hours)
8 designs (FACTS, JSON, MARKDOWN, OWN, A2, A3, G1C1, A0P) x 60 items from a separate pilot range (not the main or
sealed sets). Checks: every founder builds usable notes on most items; the JSON and Markdown repair turn works; the
own-code founders run in Docker at 20 at a time without errors; sigma is near 0.18; runs per hour at 20 workers
(this sets the real time estimate). Pilot records are never used in the results.
**Human:** read the pilot report; approve or ask for changes.

### Step 8. Pre-register and lock (human gate)
- `PREREG.md` rewritten for run 5: everything in `RUN5_DESIGN.md` section 9, the item hashes, the analysis code hash.
- `python -m v3.prereg lock --reason "V3-042: run 5"`, then `verify`; the run refuses to start if anything changed.
- **Human:** "go". Then start via `scripts/run_v3_loop.cmd`.

### Then the run itself
| Part | Runs | Hours at about 350 an hour | Human |
| --- | --- | --- | --- |
| Stage 1, generated (15 x 360) | 5,400 | about 15 | - |
| Stage 1, hand-written (15 x 500) | 7,500 | about 21 | - |
| Stage 1 report | - | - | reads it; Opus then writes the five "+" texts; human sees them |
| Stage 2 (31 x 360) | 11,160 | about 32 | - |
| Final check (4 x 720 generated + 1 x 500 hand-written) | 3,380 | about 10 | - |
| Total, with the pilots | about 28,000 | about 80 | |

Spread over two to three weeks by the weekly allowance. Stage 1 is a complete result on its own.

### Build effort (my estimate)
Steps 0 to 6: about a day and a half of building and testing in sessions, plus a few hours of Claude-plan time for
the 500 items. Then the pilot (about 1.5 hours of running), then the lock.

---

## Part B. The dashboard (streamlined, no creature pictures)

One page, five tabs. Plain text names (founder names, design numbers 1 to 32). Same server and feed pattern as now
(`v3/dashboard/server.py`, a new `feed5.py`, a new `index.html`); the run 4 dashboard is archived with the run. The
numbers refresh every 30 seconds; nothing depends on browser storage.

**Tab 1. Progress (the front page)**
- Where we are: a line of five steps (pilot, stage 1, stage 2, final check, report) with the current one marked.
- Runs done of planned, for the stage and overall; runs per hour now; estimated time left at this pace.
- Luna weekly allowance used (and when it resets); Claude plan use; judge spend in dollars.
- Failures: retried, left out, voided (with reasons); Docker sandbox calls and refusals.
- Status line: running / paused (with the reason) / waiting for the human (with what is needed).

**Tab 2. Stage 1: the founders**
- Main chart: each founder's mean with its 95% range, sorted, the control marked, founders coloured by group (notes
  only, fixed calculator, own code), with the "could be best" set shaded. Ranges narrow as items arrive.
- Table: founder, group, items done, mean, range, lead over the control (range, Holm verdict), build rate, invalid
  notes share (JSON and Markdown).
- By system size: a small chart per founder or a heat table (founder x k = 1, 2, 4, 8).
- Hand-written: the same chart for the 500, beside the generated one, and the rank agreement.
- OWN: which form Luna chose, by k.

**Tab 3. Stage 2: the five parts**
- The 32-design table (A to E as + and -) with each design's mean so far.
- Main chart: the five effects and the ten pairs, each a dot with its 95% range around zero; the seven
  pre-registered ones marked; ranges narrow as data arrives.
- The noise estimate (sigma, tau) and the half-normal plot.
- The current fitted best combination (provisional until stage 2 ends; labelled as such).

**Tab 4. Final check**
- Before: "sealed, not opened". After: the four designs on the 720 and on the 500, the paired leads with ranges, and
  the pre-registered verdicts in plain words.

**Tab 5. How it works**
- The text of Part C, with live numbers (items, runs, sigma) filled in from the feed, and a glossary with a filter.

Style: the dataviz method (thin marks, ranges always shown, one axis per chart, a table view for every chart, light
and dark modes). Hover on any point shows its numbers.

---

## Part C. How it works (the dashboard text)

Each topic is in two layers: **In plain words**, then **Technically**. This is the text the dashboard shows.

### 1. The question
**In plain words.** When an AI reads a long, messy description of a system (a factory, a power grid, a hospital
ward), it has to understand it well enough to answer questions later: what happens if, what caused this, what is
missing. We give a small, cheap AI (GPT-6 Luna) different ways to write down what it read, and measure which way lets
it answer best. A "design" is one of those ways, written as a recipe Luna follows.

**Technically.** A design is a genome: a core (the kind of notes: prose, sections, ontology, equations, rules, causal
graph, simulation, probability model, JSON, Markdown) with its parameters, writing rules, an answering procedure, an
optional worked example and a turn budget. The score is Luna's end-to-end accuracy: the share of questions answered
right from its own notes.

### 2. One run, step by step
**In plain words.** A run is one design meeting one item. Luna reads the description and writes notes the way the
design says, without seeing the questions. Then the description is taken away and Luna answers the questions using
only its notes. The answers are marked. So a design scores well only if the notes it makes really hold the system.

**Technically.** Two-phase episode (`v3/episode.py`): phase 1, Luna (Codex, allowlisted tools only) builds the
representation with the core's tools and its checker (a JSON or Markdown core must pass its format check; one repair
turn). Phase 2, a fresh Luna session gets the representation and the questions, with the core's read tools (lookup,
search, and for calculator cores evaluate or run a scenario). Generated items are marked by code against the true
answers; hand-written items by the AI judge against the answer key. One record per run, kept for audit.

### 3. The items
**In plain words.** Two kinds of test. Generated items are invented systems made by a program, so the right answers
are known exactly; they come in four sizes, from one system on its own to eight systems mixed together in one text,
so we can see whether a design copes with complexity. Hand-written-style items read like real documents (logs,
emails, meeting notes, with mistakes and contradictions) and ask open questions; they are marked by a separate AI
against an answer key. The same items are used for every design, like every student sitting the same exam.

**Technically.**
- Main set: 360 generated items (90 base systems x k = 1, 2, 4, 8 interleaved subsystems; problem families balanced),
  from seed ranges no earlier run used.
- Sealed final set: 720 more generated items, opened once.
- Hand-written style: 500 items (about 170 systems x 3 question types), written and checked by Claude Sonnet, a
  no-document leak check by DeepSeek, a human spot check; graded by DeepSeek v4.1 Flash (agreement with a Claude Opus
  referee checked first). Reported separately; never used to choose a design.

### 4. Stage 1: which way of writing things down works best
**In plain words.** Fifteen founding designs, each a different way to describe reality: plain prose (the control),
several kinds of sections, an ontology, equations, rules, a causal graph, a simulation and a probability model that
Luna programs itself, JSON, Markdown, and "Luna's own choice" with no format suggested. Every founder sits all 360
generated items and all 500 hand-written ones. We then ask which founders could be the best and which are clearly
not, allowing for luck.

**Technically.** 15 x 360 + 15 x 500 runs. Founders are compared item by item (paired), so item difficulty cancels.
The "could be best" set uses Hsu's multiple comparisons with the best (bootstrap over items for the luck margin).
Leads over the control are tested with Holm's correction. The founder with the highest mean on the generated items
goes to stage 2 even if tied; the runner-up joins the final check. Results are also shown by group (notes only,
fixed calculator, own code), by system size and by problem family.

### 5. Stage 2: what makes the winner better
**In plain words.** We take the winning design apart into five parts: how its notes are laid out, the rules for
writing each fact, how it answers, whether it shows an example, and how long it may think. For each part, one
alternative is written from scratch. Then we build every possible mix of old and new parts: 2 x 2 x 2 x 2 x 2 = 32
designs, and test all 32 on the same 360 items. Because each new part is in exactly half the designs, and balanced
against every other part, we can see what each part does on its own, and which parts help or hurt each other.

**Technically.** A full 2^5 factorial. Factor levels: "-" = the winner as it is, "+" = an alternative written by
Claude Opus from first principles (it sees the winner's genome, the task and the factor's definition; never scores
beyond who won, never items or answer keys). The five texts are frozen and shown to the human before stage 2.
Design 1 is the winner itself (its stage 1 records are reused), so 31 new designs x 360 runs.

### 6. The maths, part by part
**Comparing two designs.**
*In plain words:* for each item, take design A's mark minus design B's mark, then average. Easy and hard items affect
both equally, so they cancel.
*Technically:* paired difference; standard error = sd(difference) / sqrt(n) = 0.256 / sqrt(360) = 0.0135; 95% range
about +-0.026. Ranges come from resampling the items 10,000 times (bootstrap).

**Which founders could be best.**
*In plain words:* with fifteen founders, some gaps appear by luck alone. We work out the biggest gap luck could make
among fifteen and only rule a founder out if it is further behind than that.
*Technically:* Hsu's multiple comparisons with the best; the margin is the 95th percentile of the largest gap under
resampled items.

**Holm's correction.**
*In plain words:* if you run many tests, some say "yes" by chance. Holm's method raises the bar for each extra test
so the chance of any false "yes" stays at 5%.
*Technically:* sort p-values; the i-th smallest must be below 0.05 / (m - i + 1); stop at the first that fails.

**Measuring each part in stage 2.**
*In plain words:* the effect of a part is the average of the 16 designs that have the new version minus the average
of the 16 that do not. Every other part is in both halves equally, so it cannot pass for this part's effect. Every
run counts towards every part, which makes each part's measurement about three times sharper than comparing two
designs head to head.
*Technically:* y(d,i) = mu + a_i + sum beta_f x_f + sum gamma_fg x_f x_g + u_d + e(d,i), x = -1/+1. Var(effect) =
4 (sigma^2 / n + tau^2) / N with N = 32, n = 360; at sigma = 0.18 and tau = 0.01, standard error 0.0049 and the
smallest effect found 80% of the time is 0.014.

**Parts that work together.**
*In plain words:* a new answering procedure may only help if the notes are laid out to suit it. With all 32 mixes we
measure every pair of parts working together, as well as each part alone.
*Technically:* all ten two-factor interactions are estimated; A x C and B x C are pre-registered, the other eight
exploratory.

**How much is luck: sigma and tau.**
*In plain words:* sigma is how much a design's mark jumps from item to item. Tau is "design surprise": how much a
design scores above or below what its parts predict. More items shrink sigma's effect; only more designs shrink tau's.
*Technically:* the 16 three-, four- and five-part interactions are treated as noise and measure tau^2 + sigma^2 / n
in this run itself; t tests use those 16 degrees of freedom. A half-normal plot (Lenth's method) checks that none of
them is actually a real effect.

**Choosing the final design, and the winner's curse.**
*In plain words:* the design that scores highest is usually also the luckiest, so its score flatters it: the
winner's curse (a term from geneticists, who saw that the first study to find a gene always overstated its effect).
So we do not just take the top scorer: we take the mix the measured part effects predict is best, because those
effects rest on all 32 designs, not one lucky one.
*Technically:* the final design maximises the fitted score (five mains and ten pairs). The top observed score of 32
would be inflated by about 2.07 x sqrt(sigma^2 / n + tau^2) = +0.029.

**The final check.**
*In plain words:* one honest re-test on 720 items nobody has seen and on the 500 hand-written ones. The final design
must beat the stage 1 winner (and, separately, the runner-up) by more than luck could explain.
*Technically:* paired lead, 95% range above zero; that needs a lead of about 0.019 on the 720 and about 0.026 on the
500. Both verdicts are pre-registered and both are reported.

### 7. Safety and fairness
**In plain words.** The rules are written down and locked before the run starts, so nobody can change the test after
seeing results. The AIs that design anything never see the test questions or answers. Code Luna writes runs in a
sealed box with no internet and no access to the computer. The final test items stay locked until the end.

**Technically.**
- Pre-registration: config, code, item ids and analysis code are hash-locked (`v3.prereg`); the run refuses to start
  if anything changed; any change is a recorded decision and a re-lock.
- Separation: item writers never see designs; design writers never see items, questions or answer keys; the sealed
  720 and the v2 held-out split are refused by code.
- Luna: allowlisted tools only; any other tool call voids the run.
- Docker for model-written code: no network, read-only system, small scratch folder, memory, CPU, process and time
  limits, non-root, removed after each call; the audit-hook sandbox inside as a second layer; attack tests passed
  before the lock.
- Allowance guards pause at 90% (Luna) and 85% (Claude) and resume after the reset.

### 8. Who decides what
**In plain words.** Code runs the tests and does all the maths. AI writes the items, the alternative parts and the
notes it is tested on, and marks the open answers. The human approves the design, the items, the alternative parts
and the pilot, and says "go".

**Technically.** No AI makes a selection: the winner, the runner-up and the final design are chosen by fixed rules in
`v3/analysis5.py`. Human gates: plan approval (V3-042), founder briefs, item spot check, pilot report, lock, the five
"+" texts before stage 2.

### 9. What the run will tell us
- Which ways of describing reality work best for a small AI, with honest ranges, and whether that changes with
  complexity (k = 1 to 8) and on realistic documents.
- Whether designed formats beat what the AI does on its own (OWN), and which form it reaches for.
- Whether writing code helps (own-code group) compared with notes and a fixed calculator.
- Which parts of the best design matter, by how much, and which work together.
- One final, pre-registered verdict on sealed items.

### 10. Glossary (dashboard, filterable)
Design, founder, control (A0P), core, genome, item, k (system size), problem family, generated item, hand-written
item, answer key, judge, run, record, two-phase episode, paired comparison, 95% range, bootstrap, could-be-best set
(Hsu), Holm, factor, level (+/-), factorial, main effect, interaction (pair), sigma, tau, design surprise, fitted
score, winner's curse, sealed set, final check, pre-registration, lock, allowlist, Docker sandbox, allowance guard,
PAUSE. Each with a one-line plain definition.

---

## Part D. Human gates, in order
1. Approve this plan (V3-042, V3-043).
2. Read the JSON, MARKDOWN and OWN briefs.
3. Spot-check about 20 hand-written items.
4. Read the pilot report (mini run end to end, then the format pilot).
5. Lock and "go".
6. After stage 1: see the five "+" texts before stage 2 starts.
