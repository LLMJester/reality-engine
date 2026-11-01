# v3 decisions

Spec: `../rpp-core/reports/NEXT_ROUND_SPEC.md`. v2 is archived (`../rpp-core/ARCHIVE.md`, D-071).

## V3-001 — Launch settings (2026-10-21, human)
- **Claude usage:** builders throttled to **60 runs/hour** (`budget.claude_runs_per_hour`). The run pauses rather
  than fails at plan limits.
- **Builder model:** **Sonnet 5.5** (`models.builder_claude = "sonnet"`). Opus remains the designer.
- **Micro level dropped** (a change from spec §6, reported in the final report). A code-only level cannot judge a
  design, because building always needs an AI. Cheap exploration comes from the code-only operators (swap, simplify,
  graft, CMA-ES). The 5-item Qwen quick build test is the cheap filter (meso). Budget shares: meso 0.25, macro 0.75.
- **Grader reference:** the median of 3-4 strong graders, plus a human spot check of about 20 answers where they
  disagree most. The cheapest candidate meeting agreement ≥ 0.85 with the checked reference is chosen.

## V3-002 — Objectives used by NSGA-II (2026-10-21, implementation note)
NSGA-II ranks on e2e, buildability and −output tokens (all measured for every design). E (gold-given) and F (fidelity)
need gold models per design, so they are reported for designs where gold exists. They are not selection objectives,
because an unmeasured objective would penalise new designs. This follows D-070/KF-1: the target is buildability and
e2e.

## V3-003 — First live check (2026-10-21)
G1C3 seed on generated item W1000002:
- Qwen built a valid model and scored 0.6.
- Claude (Sonnet) answered 5/5 without building (0 tool calls).

A strong builder may skip the model, so buildability is a separate objective and the report must say when designs do
not get used.

## V3-004 (2026-10-21): frontier referee replaces the human spot check
The human could not mark 20 long prose answers. The grader-test reference is now Claude Opus (headless, no tools,
subscription), marking all ~100 sampled answers once with the v2 rubric and key points (v3/grading/referee.py). This
also removes the circularity of the consensus reference, which included each candidate's own marks. Opus graded no
answer it wrote (the sample is v2 OpenRouter answers). The chosen grader is the cheapest with agreement >= 0.85
against the referee. The human may optionally look at a few referee/grader disagreements; that is not required.

## V3-005 (2026-10-21): grader and frontier auditor chosen
Against the Opus referee (100 answers, runs/grader_test/report_referee.json): DeepSeek V4.1 Flash agreement 0.92,
consistency 0.99, $0.0012/answer; gpt-oss-120b 0.82 (below the 0.85 threshold, although it led against the
consensus); GLM 5.3 Flash 0.60; GPT-6 Luna failed every call. Grader: deepseek/deepseek-v4.1-flash. Frontier audit:
google/gemini-4-argon (a family that neither builds nor grades; works with the strict grading format, ~$0.01/answer).
Its agreement with the grader on the audited final answers is reported, not used for selection.

## V3-006 (2026-10-21): pilot findings and redesign before the lock
Pilot (state/pilot, runs/pilot_*; 3 seeds x 8 items x 2 builders + 1 generation of 6 children; 128 runs, 0 errors,
$0.68 OpenRouter, ~75 min). Results are discarded as evidence; the findings shaped the design:
- Ceiling: the plain-language control scored 0.95 with Claude (0.78 Qwen) on the full items; no design could beat it.
- Building did not help: Claude built a model in 12-38% of runs (2 turns on average); Qwen built in 75-88% of runs
  but scored below plain language (G1C3 0.72, A3 0.75 vs 0.78). Consistent with KF-1.
- Design flaw: the quick build test eliminates every child that cannot build a checkable model, so plain-language
  children always failed (one scored 0.81). Evolution could not explore plain language at all.
The human's research direction: can plain language with structure - headers and an ontology expressing hierarchy and
relationships in plain text - model complex systems better than free plain language?
Changes (all before the pre-registration lock):
1. Two buildable, checkable, evolvable plain-language cores: "sectioned" (headers, optional stated/derived/assumed
   tags, granularity; based on v2 A1) and "ontology" (controlled English: is a kind of / is part of / a small
   evolvable relation vocabulary / if-then rules / properties; checked for grammar, declared terms, vocabulary and
   hierarchy cycles; lookup, path and hierarchy query tools).
2. A complexity ladder: composite items of k = 1, 2, 4, 8 generated subsystems, interleaved without labels, script
   graded. A calibration run of the control sets the item mix so the control is off the ceiling.
3. Hypothesis H5 (structured plain language vs free plain language, and the gap against complexity).

## V3-007 (2026-10-21): two-phase episodes (build, then answer from the representation alone)
Calibration (runs/calib, single-phase, 6 composite items per k): Claude with plain language scored 0.78/0.86/0.90/0.93
at k = 1/2/4/8 and the ontology seed 0.78/0.92/0.88/0.93, but Claude made 0 tool calls in all 47 runs although the
tools were connected (transcripts: MCP "connected", add_part/check/finish/lookup/path listed): it reasoned privately
(up to ~12k thinking tokens) and said it worked the answers out by hand. Qwen with plain language fell from 0.72 to
0.28 as k grew. A single-phase episode therefore measures whether a builder bothers to build, not how good a
representation is. Change (human approved the recommendation): every episode is two-phase. The build phase sees the
description and no question and must write the representation (prose designs write plain-language notes, A0+'s
effort-matched revision passes included); a fresh answer phase sees the questions and only the representation (loaded
in the query tools, or the notes), never the description. Buildability comes from the build phase; effort is the sum of
both phases. The free plain-language control becomes "notes, then answer from the notes". Every Claude call (both
phases, every command) now passes one process-wide throttle at claude_runs_per_hour.

## V3-008 (2026-10-21): Qwen provider order and empty-response retry
In the single-phase calibration 7 of 24 Qwen ontology runs failed with an empty first response from provider Parasail
(the first in the v2 provider order), concentrated at k = 4 and 8 (long prompts). v3 now orders Qwen3 235B providers
Nebius, DeepInfra, Parasail (config models.builder_openrouter_providers) and retries an empty response up to twice.
Single-phase calibration records moved to runs/calib_single; the two-phase calibration writes runs/calib2p.

## V3-009 (2026-10-21): unfinished drafts are submitted; loose ontology gene; H6
Two-phase calibration (runs/calib2p) found that many structured builds were lost, not wrong: builders wrote every part
under the section-by-section protocol and then ended with DONE without calling finish(), so nothing was submitted
(Qwen ontology: every build; Claude ontology: 6 of 24, all "no ontology written", none a grammar error). Fix: at the
end of an episode an unsubmitted or changed draft is submitted as it stands (on a copy; marked auto_finished) and the
checker judges it. The structured rows of runs/calib2p are void for this reason; they are rerun in runs/calib2p_b
(the plain-language rows are unaffected).
The human asked whether LLMs suit loose natural language, and whether smaller models need more structure and frontier
models less. Changes: the ontology core gets an evolvable "strict" param (loose: lines outside the grammar are kept
as plain-language notes and used terms self-declare; the param operator toggles it), a loose seed ONT_LOOSE, and
hypothesis H6 (structure helps Qwen more than Claude).

## V3-010 (2026-10-21): Codex (ChatGPT subscription) as an optional third builder, under conditions
The human approved a scoped MCP approval for the arm server on five conditions; all were met before any use:
1. Restriction audit: write_model, show_model, lookup and path take text arguments only and are pure Python (no file,
   process or network calls). The server's only file I/O uses paths set by v3 (genome, state output, fixed v2 guides).
2. Allowlist enforced three times: the arm server hides and refuses other tools (--allow-tools), Codex enables only
   those tools (mcp_servers.arm.enabled_tools), and only the arm server is pre-approved (default_tools_approval_mode).
3. Everything else disabled: shell_tool, unified_exec, apps, plugins, browser and computer use, images, multi_agent,
   skills, goals, web search; read-only sandbox; empty temp working dir; user config and rules ignored; ephemeral;
   OPENAI_API_KEY removed from the environment (subscription only).
4. check_codex voids any episode with any item other than messages/reasoning/errors and allowlisted arm calls (fail
   closed: unknown item types count) and logs it to runs/integrity_log.jsonl. Designs whose tools fall outside the
   allowlist are refused for this builder (today: one_shot sectioned/ontology designs such as ONT_LOOSE).
5. Isolated tests: a probe could not read a canary file or C:/Windows/win.ini (also in the human's two manual runs);
   a two-phase ONT_LOOSE episode on K1x1901000 built a valid ontology (6 arm calls: write_model, show_model,
   write_model, 3x lookup; 179 statements, 0 problems), answered from it alone, scored 1.0, no integrity breach.
Remaining limitations: Codex still lists exec, collaboration.* (spawn_agent etc.), request_user_input and wait, which
have no feature switch in this version; exec has no file access in the tests, and any use of the others voids the
episode. ChatGPT plan usage limits apply. The model is Codex's default (not pinned).

## V3-011 (2026-10-21): Codex allowlist widened; GPT-6 Luna pinned as the small builder candidate
The human approved the full tool list for Codex: build steps (get_template, sketch, add_part, add_subsystem, check,
finish), sectioned/ontology (write_model, revise_model, show_model, lookup, path, hierarchy), equations G1C3
(submit_graph, patch_graph, show, evaluate, intervene), causal graph A2 (set_graph, add_variable, add_edge,
set_equation) and rules A5 (submit_model, patch_model, run_scenario: a fixed checker process fed structured data).
Never Codex: probabilistic (G1C1) and simulation (A3) designs, whose tools run model-written Python; refused by core,
whatever the tool names. Model pinned to gpt-6-luna (ChatGPT plan, "fast and affordable"), the free small-model
candidate to replace Qwen3 235B. Codex calls are throttled (budget.codex_runs_per_hour = 60).
Isolated tests on Luna: probe could not read a canary file or win.ini; two-phase episodes on K1x1901000 built valid
representations and scored 1.0 for SEC_A1 (section by section), G1C3 (used evaluate/intervene when answering) and
ONT_LOOSE; A0P notes scored 0.67 after a wording fix (the "use only the arm tools" line made Luna refuse to write notes
for a design with no tools; the first A0P record is kept as g8e2dd9d60dc_tools_wording_bug). No integrity breaches.
A stray surrogate in a phase-1 representation crashed the hand-off once; prompts are now sanitised.

## V3-012 (2026-10-21): Codex weekly-allowance guard
The human is on ChatGPT Pro: no five-hour Codex limit (OpenAI pricing page; confirmed by the account's rate-limit
snapshot, which has a single 10080-minute window), only an unpublished weekly cap. Snapshot at 2026-10-21T03:04:
weekly used 0%, reset 2026-10-28 02:36, plan_type "prolite", no credits. Guard: on the first Codex call and every 40
calls, one minimal non-ephemeral call ("reply OK", all tools off) reads the weekly used %; its session file is
deleted at once; snapshots are logged to runs/codex_usage.jsonl. At >= 70% (budget.codex_weekly_pause_percent),
Codex work stops (v3.run.Paused), leaving 30% of the week for the human's own use. Throttle stays at 60 calls/hour.

## V3-013 (2026-10-21): Qwen retired; GPT-6 Luna (Codex, ChatGPT Pro) is the small builder
The human asked to replace Qwen with ChatGPT. Evidence: after the finish() fix Qwen3 235B still produced almost no
valid structured representation in calibration (strict ontology 0%, loose 0-17%, sectioned 0-33% valid), while GPT-6
Luna built valid sectioned, equation (G1C3) and ontology representations in the isolated tests, at no cash cost.
Config evolution.small_builder = "codex". The quick build test, full evaluation, final evaluation and H2/H5/H6 use
Claude + GPT-6 Luna. Probabilistic and simulation designs (model-written code) are Claude-only: their quick test runs
on Claude and they have no small-builder score. OpenRouter remains only for the DeepSeek grader (v2 practice items)
and the Gemini frontier audit. The Qwen calibration records stay in runs/ as history; the Qwen strict-ontology
calibration was stopped part-way.

## V3-014 (2026-10-21): run plan approved; Claude cap 120/hour
Approved by the human: builders Claude + GPT-6 Luna; seeds A0P (control), ONT_LOOSE, ONT_BASIC, SEC_A1, SEC_FLOW,
G1C3 (all buildable by both builders; simulation/probabilistic seeds dropped, evolution can still reach other cores);
items: quick 5 composite, screening 30 composite (6/8/8/8 at k = 1/2/4/8), full 58 = 48 composite (12 per k) + 10
v2 dev items (every third dev id first, fixed); 20 children per generation, top 3 fully evaluated; stopping rule
unchanged (4 generations without improvement, max 20). The human raised the Claude cap to 120 calls/hour ("we will
see what happens"); Codex stays at 60/hour with the 70% weekly guard. The first pilot's state and records are
archived (state/archive/pilot1, runs/archive/pilot1) so the second pilot cannot reuse single-phase records.

## V3-015 (2026-10-21): micro level reinstated (on the free small builder)
V3-001 dropped the micro level because a code-only level cannot judge a design. With GPT-6 Luna free (ChatGPT Pro),
the human approved a real micro level: at the start of each generation the top 3 designs (by Claude full e2e, small
builder able to build them) each get 6 code-only tweaks (param / simplify / numeric operators); tweak and parent run
on Luna on the 5 quick items; a tweak wins if it builds on >= 3 items and its paired mean beats the parent's; the best
winner per base joins the generation's children (lineage level "micro") and then faces the normal quick test,
screening and full evaluation. Losing tweaks stay recorded (retired_micro). Micro winners earn no operator credit.
This addresses KF-2 (few, large mutations; sparse feedback). Codex cap raised to 120 calls/hour (Pro has no five-hour
limit; the 70% weekly guard remains). The unused level_budget_shares config key was removed. Note: the second pilot
started before this change, so its generation has no micro round; the dry-run test covers the micro path.

## V3-016 (2026-10-21): two-phase calibration results (pre-lock, exploratory; seeds only, 6 items per cell)
Mean score (share of valid builds) at k = 1 / 2 / 4 / 8 interleaved subsystems:
- Claude: A0P notes 0.89/0.89/0.90/0.92; ONT_LOOSE 0.78/0.89/0.86/0.93; ONT_BASIC 0.78/0.89/0.74/0.83;
  SEC_A1 0.78/0.86/0.85/0.88 (valid builds 83-100%).
- GPT-6 Luna: A0P notes 0.56/0.69/0.61/0.40; ONT_LOOSE 0.61/0.28/0.62/0.33 (valid 33-83%); ONT_BASIC
  0.33/0.44/0.47/0.25 (valid 33-83%); SEC_A1 0.61/0.58/0.28/0.37 (valid 83-100%).
Reading: for both builders free plain notes are at least as good as the seed structures; for Luna the strict ontology
is worst and the seed structures do not yet help (at seed level this points against H6). Luna has headroom at every
k; Claude is near its ceiling from k = 2. These are unevolved seeds on 6 items per cell; the run tests whether
evolution finds structure that helps. Codex weekly usage stayed at 0% after ~200 Luna calls. The item mix is kept
as approved (V3-014).

## V3-017 (2026-10-21): micro level judged on its own rotating items, with a confirmation round
Problem found in V3-015: micro winners were selected on the 5 quick items and then passed the quick test on the same
items (selection bias), and repeated use of fixed items invites overfitting. Change (human approved): a separate,
fixed 20-item micro pool (5 composite items per k, disjoint from quick/screen/full); each generation draws two disjoint
batches of 5 from it. A tweak is a candidate if on batch A it builds on >= 3 items and beats its parent (paired mean);
up to 2 candidates per base, best first, are re-tested on batch B, and the first that beats the parent again wins.
One micro round per generation (iterating was considered and rejected: compounding noise on 5-item tests, overfitting
and drift towards the small builder; macro crossover already combines winners).

## V3-018 (2026-10-21): pilot 2 passed; pre-registration locked; real run launched
Pilot 2 (state/pilot, 4 seeds A0P/G1C3/SEC_A1/ONT_BASIC, small sets) ran one full generation end to end with the
final code: seeds' full evaluation on both builders, the micro round (18 tweaks on batch A, 4 confirmation tests on
batch B, 0 winners), 6 children, quick test (4 passed), successive halving (2 → 6 items), full evaluation of 2
children, improvement test (none), NSGA-II front and operator-share update. 305 episodes, 0 errors, 0 ungraded, 0
integrity breaches; Codex weekly usage 1% after 400 calls. The micro path where a winner joins the children is covered
by the dry-run test only. Phase folders for generation 1 are named gen0 (naming only). PREREG.md text aligned with the
final config (20 children, top 3 to full, ladder item counts, deviations), then locked and the real run started
(6 seeds per config, state/state.json) under scripts/run_v3_loop.cmd.

## V3-019 (2026-10-21): template_first tools hidden from MCP clients; fix and SEC_FLOW baseline re-run
During generation 0 the integrity check voided 10 Luna answer-phase runs of SEC_FLOW (calls to list_mcp_resources).
Root cause: the MCP arm server listed only the tools callable in the current state, and for the template_first
protocol that is `get_template` alone at session start. MCP clients fetch tools/list once (listChanged false), so
`write_model` never appeared and every SEC_FLOW build was empty (Claude 43/43, Luna 30/30 unusable). Any design with
template_first was affected; no other seed uses it. Fix (approved by the human): in build mode the server lists the
arm's full tool set from the start and enforces protocol staging at call time with an informative error ("Call
get_template first"); given-gold mode is unchanged. Regression test added. SEC_FLOW's generation-0 records were moved
(not deleted) to runs/archive/secflow_template_bug/ and SEC_FLOW is re-run; the other five seeds keep their records.
The 10 voided runs are reported as integrity breaches caused by this bug. Paused 12:30, re-locked, resumed.

## V3-020 (2026-10-21): Codex resource listing allowed; voided runs excluded and reported
After V3-019, Luna still occasionally called Codex's built-in list_mcp_resources / list_mcp_resource_templates in the
answer phase when it judged the representation too thin (11 voided runs so far, all SEC_FLOW). These tools can only
list resources of the configured MCP servers; the arm server is the only one (user config ignored) and exposes none,
so the result is empty and no data is reachable. The human asked to enable all Codex tools; Claude advised against it
(shell, file and web access would expose the sealed sets and answer keys, let Luna answer outside the representation,
and break Claude/Luna parity), and the human approved allowing just these two. read_mcp_resource and everything else
stays disallowed and voided. Also decided by the human: voided runs remain excluded from scores (not scored 0), and
the final report gives each design's void rate. Runs voided before this change stay voided.

## V3-021 (2026-10-22): Claude pace raised from 60 to 90 runs per hour
The human asked for a faster Claude pace after a reset of their Claude plan allowance. config budget
claude_runs_per_hour (calls per hour; each run is a build call and an answer call) changed from 120 to 180, i.e. from
about 60 to about 90 runs per hour. Pacing only: items, grading, selection and every evaluation rule are unchanged,
and episodes are independent of the pace. Claude advised that the plan's weekly reading was 43% (reset Sunday 03:00)
and offered a weekly-usage guard like Codex's; the human declined the guard, saying the allowance had been reset.
Codex pace unchanged. Paused cleanly (state/PAUSE), re-locked, resumed.

## V3-022 (2026-10-22): generations resume after a pause instead of restarting
After the V3-021 restart the loop re-ran generation 1 from its start: evaluations resume from stored records, but the
micro round and breeding did not. The micro round skipped the 18 existing tweaks (ids already present) and made new
ones, and breeding would have asked the designer for a second set of children. Caught at 00:24 before breeding: 3
extra SEC_FLOW tweaks and 14 Luna episodes. Earlier pauses were during the baseline, which resumes correctly.
Fix (approved by the human): the engine saves the micro result and the bred children in state "pending" as soon as
each is done and reuses them on resume; a micro round paused part-way reuses each base's existing tweaks (tweaks now
record their base when made). Everything after breeding already resumed from records. Two regression tests (pause
mid-micro, pause mid-screening). Clean-up: the 3 tweaks' records and state entries moved to runs/archive/resume_bug/
(not deleted), state backed up to state/archive/resume_bug/, CMA state restored to the last numeric step before the
restart, and generation 1's micro summary and children written to "pending". The 14 episodes are excluded and
reported. Generation 1's children, quick tests and screening records are unaffected.

## V3-023 (2026-10-22): two screening tracks, Claude and Luna (from generation 2)
The human noticed that screening ran on Claude only, and so did selection (front, archive, improvement test,
finalists): evolution optimised for Claude and Luna was only measured on designs Claude favoured, which weakens H6
(a design good for Luna but average for Claude was dropped before anyone saw it). Approved by the human (option 1 of
three; the others were screening on the mean of both builders, or recording the limitation). Changes, applied after
generation 1 completed under the old rules:
- Screening: every child that passes the quick test is screened on Claude and, if Luna can build it, on Luna; each
  track runs its own successive halving (better half after 10 items, ranked on 30). Full evaluation takes Claude's top
  `full_top_k - full_top_k_small` (2) plus Luna's top `full_top_k_small` (1) not already picked; still 3 in total.
- NSGA-II objectives and the MAP-Elites archive gain a fourth objective, Luna's full e2e (e2e_small; missing counts
  as worst on that objective only). Archive occupants are compared on current objectives.
- Improvement: judged per builder against each parent and that builder's best control; a child improves if it is on
  the front and passes on either builder. Operator credit: a child wins if it beats its first parent in screening on
  either builder.
- Finalists: the front's best design on Luna is always a finalist (it replaces the last pick other than the
  structured one kept for H5).
- Catch-up (human's choice of three: catch-up by the rule, a one-off full evaluation of the best Luna quick scorer,
  or a new hand-made Luna seed): a generation screened without a Luna track gets it once, before the next generation
  breeds, by the same rule. For generation 1: its quick-test survivors are halved on Luna (10 items, better half on
  30), and Luna's top 1 not already fully evaluated gets the full evaluation on both builders and joins the front,
  archive, islands and the improvement test (an improvement resets the streak).
- Micro round bases: the top 2 by Claude plus the best by Luna (micro.bases_small = 1); previously the top 3 by Claude,
  although micro tweaks are judged on Luna.
- Codex cap raised from 120 to 240 calls/hour (Luna's work per generation roughly doubles; weekly Codex usage read 0%;
  the 70% weekly guard stays).
- Dashboard: pipeline steps (including the catch-up), review panels, How it works, glossary, and Claude/Luna toggles
  on the overview's best-design, progress and leaderboard panels.
Generations 0 and 1 used Claude-only screening; the report states this and analyses them separately where it matters.
Screening on Luna is free (ChatGPT plan) but adds Codex runs, so a generation takes longer (Codex cap 120 calls/hour).

## V3-024 (2026-10-22): state save retries a briefly locked file
On Windows, replacing state/state.json fails with PermissionError (WinError 5) while another process (dashboard,
OneDrive, antivirus) has it open for a moment; the loop crashed once (2026-10-22 00:35) and the runner restarted it.
The save now retries the replace up to 6 times over about 1.6 s before raising. Agreed in principle on 2026-10-22
02:45, deferred by the human to daytime, applied with V3-023 in the same supervised restart.

## V3-025 (2026-10-22): the catch-up runs Luna flat out
The human asked for the generation 1 catch-up (V3-023) to get through as fast as possible ("hammer Luna"). During the
catch-up only, the small builder runs 10 at a time (catchup_codex_parallel; normally 4) under an hourly cap of 1200
calls (catchup_codex_runs_per_hour; normally 240). The normal limits are restored when the catch-up ends. The Codex
weekly-allowance guard (pause at 70%) still checks every call batch; weekly usage read 0% at the change. Results are
unaffected: the same items, rule and builders, only faster. Applied with a pause and restart at about 12:10; runs in
flight at the pause are redone from the start (completed records are reused).

## V3-026 (2026-10-22): one island entry per design; tournament ties go to the score
The human asked why generation 2 bred mostly from seeds. Of its 20 bred children, 16 had a seed parent (6 from the
control A0P) and 4 a generation 1 design. Two causes were found and fixed, with the human's approval:
- **Duplicate seeds.** Generation 0's setup re-added every seed to its island each time it ran, and it ran three
  times (the pilot and reruns), so every seed was listed three times and drawn three times as often as a bred design
  in parent tournaments (simulated share on island 0: A0P 42%). Each design is now listed once: the setup skips
  seeds already on an island, the tournament counts each design once, and loading the state removes duplicates
  (recorded as an event).
- **Tie-break by id.** With four objectives nearly every design on an island is non-dominated, so tournaments were
  mostly decided after rank and crowding by the id (arbitrary). Ties now go to the higher end-to-end score on both
  builders (Claude e2e + Luna e2e), then the id.
Seeds stay eligible parents and compete on equal terms. Generation 2's children were bred before the fix (logged
here); the fix applies from generation 3's breeding. Island 2 still has no bred design until the generation 3
migration (unchanged rule). Applied with a pause and restart; completed records are reused.

## V3-027 (2026-10-22): reaching full evaluation counts as an operator win
In generation 1 no child beat its first parent on its 10 screening items, so every operator scored 0 wins. With no
wins anywhere, the share formula cut the most-tried operator hardest: swap (6 children) fell from 15% to 7%, although
its children Stout Lemur and Tidy Mole are the two best designs on Luna, and generation 2 bred no swap children.
With the human's approval, a child now also wins for its operator if it reaches full evaluation (top of its
generation's screening on either track). Micro children still earn no credit. Once, on load, the fully evaluated bred
children of finished generations get the win they would have had: repair (Clever Hornet), swap (Stout Lemur) and
swap (Tidy Mole, catch-up); shares become swap 18%, point 16%, graft 16%, repair 13%, param 12%, simplify 8%,
guidance 8%, numeric 5%, big_jump 4%. Generations 0-1 had recorded no wins, so nothing is counted twice; a flag
(credit_v3_027) stops it repeating. Generation 2's children were bred under the old shares; its credit is recorded
under the new rule. Applied with a pause and restart; completed records are reused.

## V3-028 (2026-10-22): Luna's hourly cap 480, 8 at a time
The human asked to raise Luna's hourly usage. Since the weekly reset (2026-10-22 11:42) about 2,500 Luna calls used
3% of the weekly allowance (roughly 800 calls per 1%). codex_runs_per_hour goes from 240 to 480 calls and
codex_parallel_agents from 4 to 8. The weekly guard (pause at 70%) is unchanged and still checks every call batch.
Results are unaffected: same items, rule and builders, only faster. Claude's cap (180 calls/h) is unchanged, so stages
that wait on Claude (Claude's screening track, full evaluation) speed up less than Luna-only stages (micro round,
Luna's track). Applied with a pause and restart during generation 2's full evaluation; runs in flight are redone.

## V3-029 (2026-10-23): Claude's hourly cap 360, 12 at a time, weekly guard at 85%
The human asked to double Claude's throughput. Claude's weekly allowance is the limit, not money: at 03:47 it stood at
30% (resets 2026-10-25 03:00), rising about 1.15% an hour at 180 calls/h including the human's own Claude use, and a
generation costs about 10% whatever the speed. claude_runs_per_hour goes from 180 to 360 calls and max_parallel_agents
(Claude's runs at a time) from 8 to 12, since 8 runs of about 4 minutes top out near 240 calls/h. New guard, like
Luna's: on the first Claude call and every claude_usage_check_every (40) calls, one minimal Haiku probe (the
dashboard's probe) reads the plan's weekly use, logged to runs/claude_usage.jsonl; at claude_weekly_pause_percent (85)
or when the plan reports the limit reached, the run pauses (exit 0, no automatic restart), keeping 15% for the human.
A failed probe does not stop the run. The pause event now names the guard instead of "by human". Results are
unaffected: same items, rule and builders, only faster. Applied with a pause and restart during generation 3's
screening; runs in flight are redone.


## V3-030 (2026-10-23): run 2, a clean run with GPT-6 Luna as the only builder
Run 1 ended on 2026-10-23 at 11:41 (stopping rule met). Its data is archived unchanged in
`archive/run1_2026-10-23/` (git tag `v3-run1-complete`, backup bundle in `~/rpp-backups`), with its report
(`archive/run1_2026-10-23/REPORT.md`, dev data only). The sealed sets were not opened in run 1 and are kept for run 2.
The human decided to start again from a fresh state because run 1's protocol changed mid-run (V3-019 to V3-029), and
to focus run 2 on one builder, GPT-6 Luna. This decision was made after seeing run 1's dev results (Claude near the top
of the scale, Luna still improving, the builders disagreeing); the final report must say so.
- **One builder:** `evolution.primary_builder = "codex"`. Luna builds every evolution episode (quick test, micro, one
  screening track, full evaluation); NSGA-II objectives are Luna's e2e, buildability and −output tokens; the
  improvement rule, failure targets, operator credit and finalists use Luna's results. Claude builds nothing during
  evolution (Claude Opus remains the designer for AI operators). Designs Luna cannot build (probabilistic, simulation
  cores, V3-011) fail the quick test. No catch-up.
- **Bigger population:** 100 children per generation (was 20); the top 6 by Luna go to full evaluation (was 3).
- **More full-evaluation items:** 120 (10 v2 dev + 110 composite items, 27/27/28/28 per k; was 58), to narrow Luna's
  paired CIs from about ±0.09 to about ±0.06. Quick (5), screening (30) and micro (20) sets unchanged in size.
- **Luna usage:** up to 10,000 calls per hour (in practice limited by 12 at a time); weekly-allowance pause at 90%
  (was 70%). Human's words: "as many as permitted ... i dont care to get through this quicker".
- **Final stage:** finalists chosen by Luna e2e; on the sealed sets Luna runs 3 repeats and Claude runs once, as a
  transfer check (H2, H6).
- **Unchanged:** seeds, control (A0P), stopping rule (4 generations without improvement, max 20), improvement rule,
  grader, two-phase episodes, micro level, islands and migration. Every run 1 fix (V3-019 to V3-029) is in the code
  from the start. No protocol change after launch except pause and resume.
- Estimated cost: about 3,300 Luna episodes (about 6,700 calls) per generation, about 7% of Luna's weekly allowance.
Approved by the human 2026-10-23 ("luna calls should be as many as permitted", "i want 100 children", "ok ... start").

## V3-031 (2026-10-23): transient failures retried, failed attempts kept; run 2 restarted clean
Run 2's first minutes (16:14, gen 0) hit a burst of "403 Forbidden" from ChatGPT's Codex endpoint: 15 of the control
A0P's 120 full-evaluation episodes failed and were voided (left out, not scored 0); every later episode succeeded.
- **Retry:** an episode that fails for a transient reason (a provider or connection error) is retried up to 2 more
  times, 30 s apart, within the same stage (`evolution.retries`, `evolution.retry_wait_s`). Integrity breaches,
  allowlist refusals and content flags are never retried. A failure that survives the retries stays voided and is
  reported.
- **Failed attempts kept:** when an episode runs again, its failed `record.json` is renamed `record.void<N>.json`, never
  overwritten.
- **Seeds re-evaluated on resume:** a starting design whose full evaluation has voided items is evaluated again when
  the loop starts (finished episodes come back from their records; only failed ones run).
- **Run 2 restarted from zero** (the human's choice, 2026-10-23, over fixing and resuming), so the whole of run 2 runs
  under one protocol. The partial generation 0 (16:14-16:34, about 300 Luna episodes) is archived unchanged in
  `archive/run2_aborted_2026-10-23/`. Before the restart a short pilot (`python -m v3.run pilot`, own state and
  records, results discarded) runs one Luna-only generation end to end, which had not yet run live.
- Effect on results: none on scores; voids that would have been permanent can now be recovered.

## V3-032 (2026-10-24): H1 tests one pre-named design; other finalists secondary (winner's curse)
Reviewing the tests for the winner's curse during run 2 (generation 2, 2026-10-24) found that H1 reads "the best evolved
design beats A0+", while `final.analyse` computes a CI for each of up to 3 finalists against the control without saying
which one decides H1. Counting any finalist that passes would give three chances at a lucky pass.
- **Rule:** the primary design for H1 (and H2) is the finalist with the highest dev full-set Luna e2e, recorded in the
  final lock before the sealed sets are opened. H1 passes or fails on that design alone.
- **Other finalists:** reported as secondary results; their p-values against A0+ are Holm-adjusted across the secondary
  finalists. They cannot make H1 pass.
- **Dev comparisons are descriptive:** the dashboard's "best design vs control" panel picks the top scorer of the fully
  evaluated designs on the same 120 items it reports, so its CI is biased upward and is not evidence for H1. The panel is
  relabelled accordingly (dashboard only).
- **Also noted, unchanged:** the improvement test compares a child with parents that were selected on the same full-set
  items, so the parents' scores are inflated and the test is conservative (it can hide real gains, not create false
  ones). Screening and full-evaluation item sets are disjoint (checked: 0 shared items), so a full-evaluation score
  carries no selection bias from screening.
- **Effect:** no change to evolution, items, scoring, finalists or the sealed-set runs; only the reading of the final
  result, which becomes stricter. No code change: `final.analyse` already computes every number needed. The loop
  checks the lock only at start, so the re-lock needs no pause.
- No sealed data has been seen. Approved by the human 2026-10-24 ("yes do both").

## V3-033 (2026-10-24): H7, the best design holds up better as systems get more complex (human's hypothesis)
A read-only look at run 2's dev data (generation 2) by complexity level found that Velvet Owl (gf968a669755, simplify
of SEC_A1, the best design so far) has its lead concentrated in the most complex systems:
- Luna e2e by k = 1/2/4/8: Velvet Owl 0.704 / 0.636 / 0.565 / 0.531; SEC_A1 0.728 / 0.667 / 0.524 / 0.429; A0P
  0.630 / 0.500 / 0.539 / 0.411. At k = 8 Velvet Owl is the best of all 12 fully evaluated designs (next 0.429).
- Difference vs A0P at k = 8: +0.121 [+0.055, +0.183]; at k = 1: +0.074 [-0.123, +0.259]. Vs SEC_A1 at k = 8: +0.103
  [-0.027, +0.231]. The fitted trend over log2(k) vs A0P is flat (+0.003 [-0.062, +0.066]); k = 4 breaks the pattern.
- Caveats: Velvet Owl was chosen as the best on these same items, and the data were cut five ways, so this is a
  hypothesis, not a finding.
The human's argument: the winning design is a better way to describe a system that is built up over many interactions
with an AI builder (enhancements, tweaks), because it keeps a better grasp of complexity.
- **Rule (H7, secondary):** for the H1 primary design (V3-032), on the sealed generated composite items with the Luna
  builder, (difference vs A0+ at k = 8) minus (difference at k = 1) has a paired cluster-bootstrap 95% CI above 0.
  Reported for Claude and per k. A k = 8 vs k = 1 contrast is pre-specified, not a slope, because the dev pattern is
  not a smooth trend. H5's existing secondary slope test is unchanged.
- **Scope:** H7 tests one-shot questions about systems of growing size. The human's wider claim, that the design stays
  better as a system is built up and changed over a conversation, is not tested by H7; it belongs to a separate
  follow-up study after run 2 (multi-turn conversations with dependent follow-ups, fresh items outside the sealed seeds).
- **Effect:** none on evolution, scoring, finalists or the sealed runs; no code change (computed in the report from
  per-item records). The loop checks the lock only at start, so no pause. No sealed data has been seen.
Approved by the human 2026-10-24 ("ok write it down").

## V3-034 (2026-10-24): run 3, directed breeding toward one goal, with built-in finals
Run 2 was paused by the human on 2026-10-24 at 08:48, in generation 2, after generation 1 found no improvement (best:
Velvet Owl gf968a669755, Luna full e2e 0.586, not clearly ahead of its parent SEC_A1 0.561). The human asked why the
best designs never bred with each other: with three islands, binary tournaments and only swap/one_point using two
parents, 1 of generation 2's 101 children had two parents, and the top pairs on each island were parent and child.
A cross-breeding probe (`experiments/cross_probe_2026-10-24/`, separate from run 2, 40 fresh composite items, Luna)
found: the pre-named cross (Velvet Owl + Merry Dunlin's guidance vs Velvet Owl) failed, -0.006 [-0.140, +0.118]; the
other two targeted crosses were the top designs (0.675 and 0.670 vs the control's 0.547; each clearly above the
control, at the edge of clearly beating both parents); and Velvet Owl's dev lead over A0P (+0.083) shrank to +0.029
on fresh items, the winner's curse seen live. The human then decided (2026-10-24): "i want to breed a winner",
"optimize breeding for best outcomes, not mimic actual evolution", archive run 2 as it stands and design run 3 together.
- **Decisions, asked one at a time:** goal = overall Luna score; start from scratch; children mostly targeted (about
  50% fixes, 30% combinations of proven changes, 20% random: combining leaders "should be part of the breeding", not
  always chosen); 30-40 children per round; grow items until proven, then win on a fresh set (120 items); one pool
  with a small reserve; stop after 4 rounds without a proven gain; archive run 2 as it stands.
- **Goal statement (human):** a creature that describes complex systems with the highest accuracy for interactions
  with AI and for building systems; the winner is then tested by finals built into the run: accuracy on the sealed
  sets, an interaction/build test (the multi-turn study proposed with H7, now part of run 3), and a "super final" on
  other small OpenRouter models. No Claude in run 3 at all ("we need a separate eveolution for the best frontier
  creature"); Claude Opus remains the designer only (confirmed: "opus designer yes").
- **Seeds:** after an online search on how to write descriptions for AI (prompt-format sensitivity, strict formats
  hurting reasoning, fact lines and key-value lists, entity-state tracking, program-aided reasoning, EARS specs), the
  human reviewed every seed: kept A0P, SEC_A1, ONT_LOOSE, SEC_FLOW, G1C3; added back A5 (rules); added FACTS, STATES
  and SPEC (sectioned cores, no new code); dropped ONT_BASIC. Provenance with sources: `docs/seeds_run3.json`, shown
  in the dashboard's Details tab. Seeds display as "Seed <Adjective Animal>" with their code name.
- **Mechanics (code: `v3/directed.py`, config `evolution.mode = "directed"`, `evolution.directed`):** as PREREG "Run 3
  rules": leaders as parents, fix / combine / explore kinds with adaptive shares inside each kind, a transplant
  operator (`deterministic.transplant`), a reserve from other families, one pool, no micro, a champion changed only by
  a growing proof (120 -> 240 -> 360 items, CI above 0 vs champion and control) plus a fresh 120-item confirmation.
  The designer now sees the goal and parent 1's own failure profile (`Directed.failure_profile`; checker problems via
  `evolve.build_problems`, quoted item terms blanked).
- **Graft fixed (raised in run 2):** graft is never offered while the gene library is empty, and every failed attempt
  to make a child counts as a try for its operator, so an operator that cannot apply loses share instead of keeping
  its prior.
- **Finals (`v3/final3.py`, config `final`):** run automatically after the stopping rule (`directed.auto_final`).
  A: sealed sets on Luna, 3 repeats; H1 pooled over the sealed sets (runs 1-2's H1 used the v2 held-out set alone; run
  3's evolution score pools v2 and composite items, so the test does too), H7, H5, H3, H4. B: 20 four-turn build-up
  chains on the test split, H8 and its compounding contrast. C: four small OpenRouter models (Qwen 3.8 Flash, Gemma 4
  26B, Mistral Small 2603, gpt-oss-20b; spare Nemotron 3.5 Lightning), proposed by Claude from OpenRouter's model list
  (small, cheap, tool use, four vendors, released at least two weeks earlier), H9. Estimated OpenRouter cost of final
  C about $2-3; v3's OpenRouter spend so far is about $4 of its $35 limit.
- **Unchanged:** two-phase episodes, grader, quick test, successive halving, transient retries (V3-031), the Luna
  allowance guard, the control.
- **White paper:** the human wants a paper "Describing Reality for AI" at the end, with the winning creature in an
  appendix so others can test it; final3 writes a paper kit for it.
- Estimated cost per round: about 2,000-2,500 Luna episodes (36 children: 180 quick, about 860 screening, 480 full;
  extension and confirmation runs for candidates, the champion and the control).
Approved by the human 2026-10-24 ("approved, 120 items, propose the models, opus designer yes"; the seeds one by one;
A5: "add and update the tests").

Addendum before launch (2026-10-24, approved by the human: "Start now, fix C later"): the full run 3 pilot was cut
short in round 1 at the human's request ("seems wasteful of resources, would rather run the experiment") after it had
shown the seeds building on Luna (all 9; on 10 pilot items SEC_FLOW and STATES 0.744, FACTS 0.705, A0P 0.582; STATES
and ONT_LOOSE build 60%), round 1 breeding the planned mix (5 fix, 3 combine incl. 2 transplants, 2 explore), and the
quick test, screens and full stage. Instead of the finals pilot, a quick check ran one A0P episode per transfer model:
only Mistral Small worked cleanly (0.667, 6 of 6 answered). Qwen 3.8 Flash, gpt-oss-20b and Nemotron used their whole
12,000-token output on reasoning; with reasoning effort "low" gpt-oss and Nemotron ran but missed the answer format,
and Gemma and Qwen ran out of tokens. `v3/llm.py` gained an inactive hook (`final.transfer_reasoning`, not set) for
per-model reasoning settings, and final C runs only when `final.transfer_ready` is true (false at launch). Final C's
settings are left to V3-035, fixed on validation items with seed designs only before the stopping rule and before any
final C record; if it is not locked when the finals start, final C is skipped and reported as not run. Records:
`runs/smoke_transfer/`, `runs/smoke_transfer2/`, `state/pilot/`, `runs/pilot_gen0_*`.

## V3-035 (2026-10-24): final C settings, Claude Haiku as a fifth transfer model

What: final C (transfer) is made runnable. (1) Per-model reasoning settings (`final.transfer_settings`, applied in
`v3/llm.py`): Qwen 3.8 Flash reasoning off, gpt-oss-20b effort low, Gemma 4 26B and Mistral Small none. (2) Final C
reads answers leniently (`items.parse_answers(lenient=True)`, used only for `or_*`/`cl_*` builders): also
`FINAL ANSWER {...}` and bare `s<j>.p<n>: <option>` pairs after the last marker; the evolution and finals A and B keep
the strict `ANSWERS:` reader. (3) Claude Haiku added as a fifth model (`cl_haiku`: Claude Code headless on the human's
subscription, the run 1 Claude builder with model `haiku`, under the Claude hourly cap and weekly guard); evolution
stays Claude-free. (4) No spare; a model with paired champion/control records on fewer than half of the 60 items is
reported as unusable and left out of H9 (`final3.h9`); H9 needs at least 2 usable models. (5) `final.transfer_ready`
true.

Why: the pre-launch check (V3-034 addendum) failed most models. Cause 1: unbounded reasoning used the whole 12,000-token
output (Qwen, gpt-oss, Nemotron); setting effort "low" for every model made Gemma and Qwen think and run out instead.
Cause 2: A0P's own system prompt (from v2) asks for a FINAL ANSWER line while the composite question asks for an
ANSWERS line; Luna copes, small models followed the A0P prompt ("FINAL ANSWER {json}", or bare pairs), scoring 0 with
correct-looking answers. The human asked for Haiku on the subscription ("i meant a haiku sub agent on my subscription
for the final c") and approved: "add it as a fifth model test it and if other models cannot be used like mistral for
whatever reason that is fine".

Numbers (validation items K2x1975000 and K4x1975000, A0P and SEC_FLOW only; `runs/smoke_transfer3/`,
`smoke_transfer4/`, `smoke_transfer_haiku/`): Qwen 4/4 answered (0.33-0.67); Gemma 4/4 (0.58-0.83); gpt-oss 4/4 with
the lenient reader (0.00-0.50); Haiku 4/4 (0.83-0.92, 1-3.5 min each); Mistral 1/4 ran (0.67), 3/4 rate-limited
upstream (429, Mistral's shared pool, also earlier in the afternoon). Rejected: Nemotron 3.5 Lightning (answered
"NO CHANGE", empty responses), GLM-5.3 Flash (reasoning cannot be disabled on that endpoint), Granite 4.2 8B (ran out
of tokens, 20-40 min episodes).

Effect on results: none on evolution, round 0 or finals A and B (strict reader and Luna unchanged; verified by
tests). Final C now runs, on up to five models. No final C or sealed data existed when this was decided (round 0 in
progress). Applied by a pause and runner restart during round 0 (runs in flight finish; resume from records).

## V3-036 (2026-10-25): growing proof on fresh items only; the round's best child always meets the champion

What: (1) the growing proof judges only on the extension sets (fresh items no design was selected on): ext1 (120),
then ext1+ext2 (240), then ext1+ext2+ext3 (360) while undecided; the full set is no longer part of the proof. (2) A
third extension set ext3 is added (30 bases x k = 1, 2, 4, 8, bases 1,669,500-1,688,500, next in the same fixed
sequence as ext1 and ext2, which are unchanged; added to the running state's item sets by `v3.run.evolution` after
checking every other set is identical). (3) Each round, the best fully evaluated child enters the proof whatever its
full-set score (`directed.proof_best_always`), in addition to any child above the champion's full score. Proven,
the fresh confirmation (120 new items, never reused; mean above champion and control) and at most one confirmation
per round are unchanged.

Why: the human noticed that the starting champion (FACTS, Fuzzy Dunlin) never went through the proof itself: it was
the best of nine founders on the same 120 items, so its 0.642 carries selection luck, and challengers had to beat that
inflated score just to be compared. Round 1 showed the noise: 10-item and 30-item screening scores correlated only
0.37, the survivors' mean fell from 0.72 to 0.57, and Ember Viper went 0.69 on 30 items to 0.613 on 120 (below 0.642,
so it would never have met the champion). Judging the proof on the full set also kept the champion's luck in the
comparison. The human: "from the full evaluation the highest should face the current champion regardless of whether
it is better"; approved with the third extension set ("yes do both with the third extension set"). Re-testing the
founders was considered and declined by the human ("its too late to test the founders... growing proof will pit the
best evolved against a fair test with the seed winner").

Numbers: cost per round about 120-360 extra challenger episodes; the champion and the control meet each extension set
once (about 720 episodes in all, the first time a proof reaches each set). Power: 360 paired fresh items detect gains
of about +0.05 to +0.06.

Effect on results: from round 1's proof step on, every evolved round uses the same rule (round 0 had no proof). Proofs
become fairer to challengers (the champion's selection luck is out of the comparison) and to the champion (the
challenger's own luck on the full set is out too). One extra proof per round slightly raises the chance of a lucky
proof; the fresh confirmation and the sealed finals guard against it. Applied by a pause and runner restart during
round 1's full evaluation, before its proof step.

## V3-037 (2026-10-25): run 4, honest screening, report cards, smart combining, family places, 8-round stop

What: a new run (run 4) with its own rules ("Run 4 rules" in PREREG.md; code `v3/directed4.py`, config
`evolution.mode = "directed4"`; run 3's code path is unchanged and still tested). In short:
(1) Screening: one ordered 120-item main set, interleaved so every stage covers k = 1, 2, 4, 8; scores add up (each
item run once per design and reused); the 5-item quick test is pass/fail only; screen 1 = 35 items, screen 2 = 65,
cutting only children whose paired 95% CI against the current champion on the same items is entirely below 0; the
top 6 go to the full 120 (one place held for another family). (2) The AI designer gets a report card per parent:
score by size, build rate, and wrong answers with its own representation on 8 practice items that never score
designs. (3) Mix: 20 children, ~30% fix / 35% combine / 35% big ideas, with a new answer_method operator (how Luna uses
its notes to answer); repair only for parents whose builds still fail. (4) Smart combining (each gene taken whole from
the parent that carries the strength, a reason per gene, bad mixes fed back) and a near-winners shelf (proof within
0.02 of the champion) used for half the combines; run 3's near-winners Stormy Marten, Lanky Finch, Little Marten and
Lucky Beetle join round 0. (5) About 1 in 5 children are reserve children of other families; family-share alarm at
70% for two rounds. (6) Stop after 8 rounds without a new champion (cap 20). All item sets are new (main from
validation 1,300,000; ext from train 100,000; practice train 300,000; confirmation train 500,000; pilot validation
1,450,000) and checked against every item used before (`scripts/check_item_ranges.py`: no overlaps). Fresh sealed
sets (generated test seeds from 2,100,000, composite ladder from 2,600,000, interaction from 2,800,000); run 3's sealed
sets and the v2 held-out set are retired. Dashboard: run 4's pipeline steps, renamed "Describing Reality for AI", and
creatures that show sectioned/ontology bodies and guidance markings (most children differ only in guidance).

Why: run 3 ended after 4 rounds without a new champion; every round's best child came within 0.6 to 1.9 points of
the champion on fresh items. The human found the mini tests "pointless" and asked for better breeding. The offline
replay of run 3 (`paper/side/replay_screening.txt`) showed: the 10-item screen held only k = 1-2 items (the first 10
of a size-sorted list); re-running the same child on the same 10 items moved its mean by 0.084 on average (episode
noise) while children differed by 0.071 (SD); stage-to-stage correlations were -0.11 to 0.47. The designer mostly
fixed build failures although builds already passed (18 fix children per round), and every design fell from about
0.8 at k = 1 to 0.53 at k = 8. Each improvement was explained and approved one by one by the human on 2026-10-25
("yes good, proceed"), including the adjusted screening and "the next run cant be dunlin its who ever wins from the
r0" (the screens race the current champion). Smart combining follows the human's "when they are combined strenghts
must be used not the weak parts".

Numbers: round 0 = 13 starting designs x 120 items (1,560 episodes); each round about 100 quick + up to 600 + up to
600 screening + 330 full + practice runs for new parents (8 each) + the growing proof (120-360 per candidate, the
champion and control once per extension set): about 1,300-1,700 episodes, a little more than run 3's ~1,300. Power: on
35 items a clear loser is about 4+ points behind; on 65 about 3.

Effect on results: run 4 is a separate experiment; run 3's results stand as recorded. Screening decisions become
less noisy and never discard a child that is close to the champion; the champion still changes only by the strict
growing proof and fresh confirmation. Run 3's finals were not analysed before run 4 was designed.

## V3-038 (2026-10-25): nine genes (guidance split) and three-parent combining, before run 4's pilot

What: (1) the guidance gene is split in three: guidance (how the builder writes its representation), answering (how it
answers questions from the representation) and examples. `schema.upgrade` gives pre-split designs an empty answering
gene and moves their examples to the examples gene; `schema.legacy` maps back, and the id is the hash of that pre-split
form, so every earlier design (seeds, run 3's carry-overs) keeps its id and renders exactly the same builder prompt.
The answering text is added to the builder's guide under "When you answer questions from the representation". The
guidance operator may change guidance and/or examples; answer_method may change only answering and the query layer.
(2) smart_combine3: 2 combine slots per round (config `directed.three_parent`; not shelf slots) take each gene whole
from one of three parents, the best fully evaluated design of three of the up-to-four strongest families among the
leaders, the shelf and the champion. Every parent must give at least one gene that matters (the plain-language cores
ignore the schema gene) and whose value only it has; the designer may reply that no such mix fits; after two tries
the slot becomes an ordinary combine. The two-parent smart combine also needs a gene that matters from each parent.
Bad mixes are judged against all parents. Dashboard: 9 genes, the new genes' values, creature parts for them (answering:
ears, crest, brows; examples: beads and cheeks), the number of three-parent children in the breed step.

Why: the human asked whether 7 genes are enough. For the plain-language designs that lead, only about four genes do
anything (core, build protocol, guidance, numeric), and the guidance carried almost all ideas: in run 3, 56 of the
single-parent children changed only the guidance and 48 of 129 designs used examples. Combining could not join one
parent's way of writing notes with another's way of answering, because both sat in one gene. The human approved the
three-parent combine ("yes add 3 parent combine, 1-2 per round") and then the split ("do the split and other
amendments and quick test before the pilot that you think are meaningful").

Quick test before the pilot (scratchpad smoke, phase `runs/smoke_v3038`, an unused train item K4x900000; never used
for scoring): a live designer call for answer_method returned only an answering gene (a six-step procedure: scope,
collect, candidates, trace, decide, check); live smart_combine3 calls first showed a parent giving only genes all
three shared, which led to the "only it has" rule; with that rule the designer replied that no mix of the starting
designs fits (they differ in few genes and none has an answering gene yet), so early three-parent slots will mostly
become ordinary combines. Two Luna episodes ran without errors (FACTS 0.67, FACTS with the new answering gene 0.92 on
one item: an anecdote, not evidence).

Numbers: no change in episode counts (the three-parent children replace two combine children).

Effect on results: none on runs 1-3 (ids and prompts unchanged; their code path still tested). Run 4 starts with the
split genes; applied before its pilot and lock.

## V3-039 (2026-10-25): run 4 drops the four designs carried over from run 3; round 1 restarts from the nine seeds

What: config `directed.carryover` is set to null (the file stays on record as `carryover_dropped`), so
`carryover_seeds()` returns nothing and the run's starting designs are the nine seeds. The state is rebuilt with the
same item sets; round 0 is recomputed by the baseline from the stored records of the same 120 main-set items (no new
Luna runs), which makes the best of the nine the starting champion. Round 1's first breeding (17 children, a few quick
tests) and the carry-overs' practice records are moved to `archive/run4_round1_v3_039/` unscored; the old state is
backed up in `state/archive/v3_039/`. The pilot code keeps its two carry-overs (read from `carryover_dropped`).

Why: round 0 made Stormy Marten champion, and five of the six leaders were FACTS and its near-copies: Stormy Marten
differs from FACTS only in the numeric gene, Little Marten and Lucky Beetle only in the core, Stormy and Little in core
and numeric. Breeding had no diversity to work with: smart combining failed all 10 attempts (a child must take a gene
that matters from each parent and differ from both), the leaders held only two families so no three-parent trio was
possible, 3 of the 20 children could not be made, the 4 combines that were made were code transplants between
near-copies, and the champion and FACTS got 1 and 0 children. The human chose to drop the carry-overs ("option 1")
over keeping them with one leader per family.

Numbers: no extra runs (round 0 reused; round 1 is bred again). Round 1 restarts about an hour later than planned.

Effect on results: runs 1-3 unaffected. Run 4's round 0 scores are unchanged; the carried-over designs' scores are
reported but they no longer compete or breed. The starting champion becomes the best of the nine seeds on the same
items (FACTS, 0.625). Nothing from round 1's first breeding is scored or used. Applied 2026-10-25 after round 0, with
the loop paused.

## V3-040 (2026-10-25): from round 2, every round breeds a fix child and an answer_method child from the champion

What: config `directed.champion_children_from_round: 2`. In `Directed4.breed`, from that round on, the first fix slot
and the first big-idea slot that are not reserve slots take the champion as parent: the fix slot draws its operator as
usual (point, guidance or repair, from the report card), the big slot is answer_method. Each keeps the champion for two
tries, then breeds as usual. `breed_one` gained a `parent` argument. Test:
`test_champion_gets_a_fix_and_an_answer_method_child_from_round_two`.

Why: parents are drawn by rank weight (the champion about 28% of single-parent slots), and each round's draws come from
a fixed seed. Round 1's draw gave the champion none of its 13 single-parent children twice running (Stormy Marten one
child before V3-039, FACTS none after), so no child worked on the champion's own report card or tried a new way of
answering from its notes; in the pilot that kind of child (an answer_method child of FACTS) was the strongest. The
human approved the guarantee ("yes add the champion guarantee from round 2").

Numbers: no change in children per round (2 of 20 slots are directed to the champion) or episodes.

Effect on results: none on runs 1-3 or on run 4's round 0 and round 1 (it starts at round 2). From round 2, the
champion is the parent of at least two children per round; the remaining slots and all tests are unchanged. Applied
2026-10-25 during round 1, by a restart through the runner; round 1 resumes from its stored records.

## V3-041 (2026-10-28): run 4 stopped by the human and archived; the white paper on hold; run 5 to be a designed experiment

What: run 4 was paused at 11:35 on 2026-10-28 during round 11 (quick checks only) for a side test, then stopped for
good. Its records and state are moved unchanged to `archive/run4_2026-10-28/` (README, REPORT; git tag
`v3-run4-archived` is the repository just before the move). `state/PAUSE` stays in place: there is no running
experiment until run 5 is built and locked. The usage and spend logs stay in `runs/` (copied to the archive) for the
weekly Codex guard. Run 4's finals were not run and its sealed sets stay unopened.

Why: the human: "i am not sure i will learn anything more from run 4"; "relying on chance to pick the winner is
silly"; "nothing novel has taken the lead in run 4, its basic facts all the way". After round 10 (streak 4 of 8), the
best children were about +0.015 better than Zesty Lynx, too little for a 360-item proof to show, and the top 12
designs differ no more than item noise predicts (estimated true spread 0.000, `paper/side/run5_noise/noise.json`).
Two side tests while paused (Hand of God I and II, `paper/side/`) found that the run's data is confounded (genes
travel together down family lines), so even a mathematical pick could not rank near-equal designs.

Decisions (human, 2026-10-28): archive run 4 as it stands (no more rounds, no finals); put the white paper on hold, to
be rewritten around run 5 "as if run 4 never existed", keeping its format and the results that still apply; drop the
Hand of God slot; run 5 tests a set of designs chosen in advance, every design on the same items (about 360), 20
Luna runs at a time, with a streamlined dashboard without creature pictures. Draft: `docs/RUN5_DESIGN.md` (a
resolution V half fraction of five two-level factors, 16 designs plus the control, analysed by a pre-registered
linear model; one final check on fresh items).

Effect on results: runs 1-3 unaffected. Run 4 ends without a final verdict: Zesty Lynx (ga4fe9e97117) is its last
champion, not a tested final. Anything reported from run 4 must say it was stopped early by the human after seeing
round 10's results.

## V3-042 (2026-10-28): run 5 approved: a designed experiment in two stages

What: run 5 is built as `docs/RUN5_DESIGN.md` (all six questions decided by the human on 2026-10-28) following
`docs/RUN5_BUILD_PLAN.md`. Stage 1: fifteen founders (the nine, A2, A3, G1C1, JSON, MARKDOWN, OWN) on the same 360
fresh generated items and 500 new hand-written-style items. Stage 2: the full 2^5 factorial (32 designs) on the parts
of the stage 1 winner, on the same 360 items. Final check: the final design, the stage 1 winner, the runner-up and
the control on 720 sealed items, and the final design on the 500. Luna (GPT-6 Luna through Codex), 20 runs at a time.
A new run engine (`v3/run5.py`) and analysis (`v3/analysis5.py`) replace the breeding loop for run 5; run 4's code
stays in place, unused. The Codex allowlist is widened for the new cores' tools (JSON and Markdown lookup and search,
and the sandboxed code tool), approved with this plan.

Pilot (human, 2026-10-28: "your pilot needs to include a mini run with say 100 test of the entire process to check
everything works end to end"): besides the 480-run format pilot, a mini run of about 100 runs goes through every
stage on pilot items (stage 1, the automatic choice of winner, writing and freezing "+" texts, all 32 stage 2
designs, the sealed final check, the analysis and the dashboard). Its records are never used in the results.

Why: the human: "relying on chance to pick the winner is silly"; "just run everything through 340 tests or whatever,
20 at a time"; "first principles"; "i also want to include markdown and JSON structure"; "yes go ahead, start
building" (2026-10-28).

Effect on results: runs 1-4 unaffected. Run 5 starts only after the pilot, the human's checks and a new lock.

## V3-043 (2026-10-28): model-written code allowed for Luna, in Docker only

What: the Codex builder's refusal of the simulation and probabilistic cores (`REFUSED_CORES`, V3-011) is lifted for
run 5, so founders A3 and G1C1 can run. Every program Luna writes runs in a throwaway Docker container: no network,
read-only system, a small scratch folder, memory, CPU, process and time limits, non-root, no added privileges,
removed after the call, with the v2 audit-hook runner inside as a second layer (`v3/sandbox.py`). Attack tests
(network, files outside the folder, the host disk, other programs, fork bombs, memory, endless loops, disk filling,
secrets in the environment) must all fail before the lock. If Docker is not running, these runs wait; they never
fall back to running code on the host.

Why: the human: "it has to understand the english to write the code"; "yes use docker" (2026-10-28).

Effect on results: adds two founders to stage 1 (the "own code" group). A description that needs code to be useful
carries over less well to settings where an AI can only read text; the paper says so.

## V3-044 (2026-10-29): run 5 design revised after an independent review, before the lock

What: the human had the plan (`docs/RUN5_REVIEW_BRIEF.md`) reviewed by Gemini Argon (high thinking) and approved all
the changes recommended in reply ("approve all adoptions and go to 480 items"). Before the lock:
1. Thinking budget (factor E) and compute: when the final design has the doubled budget, the sealed check also runs
   the winner with only the budget doubled (design 17), and the lead is split into a recipe part and a compute part.
2. Stage 1 is described as testing representations together with their tools; a pre-registered contrast compares
   the notes-only founders with the tool-assisted ones (fixed calculator and own code).
3. Failures: a failure caused by the design (a forbidden tool call, the episode running out of time) is retried twice
   and then scored 0 (zero.json beside the record), never dropped; an infrastructure failure is retried with backoff
   (60 s, 300 s) and then halts the run for the human (the pair runs again after the restart). Replaces "left out
   and reported, never scored 0" (from runs 1-4).
4. Stage 2 runs all 32 designs fresh in their own record folder (runs/run5_s2); design 1 no longer reuses the stage 1
   winner's records (they carry the winner's luck).
5. Stage 2 inference: the primary tests are across base systems (per-item contrasts averaged per base, t test on
   bases - 1 df, cluster bootstrap range); the 16-df fit on the 32 means is secondary. Pre-registered tests: the five
   parts, A x B, A x C, B x C and A x B x C (nine, Holm). The final design is the argmax of the fitted model with
   five mains and ten pairs, plus A x B x C if its primary p < 0.05.
6. Precision table corrected (the pilot's paired sd is 0.36, not 0.256; within-base correlation 0.03, so the design
   effect is about 1.1, not the reviewer's assumed 2): 80%-power leads about 0.049 head to head on 480 items, about
   0.040 in the final check on 720.
7. If the founder with the highest hand-written mean is neither the winner nor the runner-up, it joins the final
   check.
8. A BLIND baseline (Luna answers with no description and no notes; no build phase) runs on every stage 1 item,
   beside the founders, never chosen. Hand-written leak check tightened: the no-document answer must score at most
   0.25 (was below 0.5) and must not hit the item's highest-weight key point (partial credit from general reasoning
   allowed: requiring a miss would have rejected 37 of 186 checked items whose no-document score was mostly 0).
   60 top-up systems (H181-H240), each with a diagnosis question; the 500 are chosen by quota per question type.
9. Judge calibration: after stage 1, 100 hand-written answers (spread over the founders, fixed seed) are re-graded by
   Claude Opus with the same rubric; agreement and per-founder bias are reported (not gating). The rubric already
   counts hedged wrong alternatives as errors and ignores length.
10. Final check in order: confirm 1 = stage 1 winner beats the control on the sealed items; confirm 2 = the final
    design beats the winner, only if they differ.
11. The human may send the + texts back only for a written reason (wrong factor; refers to tools the design lacks;
    mentions items, answers or results; breaks the length rule), at most twice; B's length must be 0.5 to 3 times
    the winner's (at least 600 characters allowed).
Also: main set 120 bases x 4 k = 480 items (was 90 x 4 = 360); ties on the generated mean broken by the hand-written
mean, then by name; mean output tokens and wall-clock time per design, and accuracy conditional on a passed format
check, reported as secondary outcomes.

Not adopted: running a half fraction (16 designs) on both of the top two founders. It conflicts with the full 32
designs the human chose, and it aliases A x B x C with D x E. Change 7 covers a near-tie at the top.

Why: the human: "approve all adoptions and go to 480 items" (2026-10-29).

Effect on results: nothing has run in run 5 yet (only the pilots, whose records are never used). Budget about 35,000
runs (stage 1 16 designs x 980 items; stage 2 32 x 480; final check 4 to 6 designs x 720, plus 500).

## V3-045 (2026-10-29): fixes from an independent code review (Gemini), before the lock

What: the human had the run 5 code (`review/run5_code_for_review.md`, the version before V3-044) reviewed by Gemini.
Each of its 24 findings was checked against the current code. Two were not bugs: (1) the JSON and Markdown cores do
show Luna the check problems with line numbers (the rendered arm adds them, as for every checked core); (6) model
code does run in Docker (v3.genome.render installs the sandbox whenever an own-code design is rendered). Four were
already fixed by V3-044 (failures, backoff and halting, quota selection, stage 2 inference). Fixed now:
- Cores: a template placeholder must start with a letter (or be <...>), so comparisons such as "T1 < 10% and
  P1 > 50" are no longer rejected (sectioned and the new cores alike); JSON lookup matches like Markdown lookup
  (all words, in a key or a naming field); an empty lookup path no longer returns the whole object.
- Stage 2 builder: the worked example is written in both layouts and each design shows the one matching its layout
  (no artificial A x D clash); the new fact-writing rules may not name sections or keys (like the answering
  procedure); the new layout must keep any section names the winner's own rules mention; every level must differ
  from the winner's (E+ is half the budget if doubling passes the maximum); the 32 designs must be distinct; layout
  names are checked in any case next to words like "section"; the frozen hash covers the winner, and the engine
  checks the + texts were written for this winner.
- Analysis: the could-be-best set uses a studentised one-sided margin per founder (Hsu against the best other),
  not the all-pairs range; one-sided p-values use the percentile bootstrap, consistent with the ranges; Spearman uses
  average ranks for ties; stage 2 refuses an empty item set.
- Engine: a stage never hands over with a missing score (halts instead); a founder that is the final design is
  compared under one label; stale pause and halt reasons are cleared.
- Sandbox: program output is capped while it is read (a flood stops the program), so it cannot exhaust memory.
- Items: the leak check is answered by Claude Sonnet (a model at least as capable as the writer) instead of the
  judge model grading its own answer, and is re-run with Sonnet on every item checked earlier (the DeepSeek result
  stays on record); a grade that fails is retried, never saved as a rejection; a rewrite first writes the
  adjudicator's key fixes back and restores the system if the rewrite fails; JSON in replies and simulator output is
  read robustly; the writer must print an entry for every item and positive key-point weights.
- Smaller: the OWN form classifier needs real program syntax to call notes code; the item-set choosing switch is per
  thread; the dashboard's live stage 2 needs items common to all 32 designs.

Why: correctness before the lock; the human asked for the code to be reviewed and sent the review.

Effect on results: nothing in run 5 has run (pilots only). The placeholder and lookup changes alter what the JSON,
Markdown and sectioned founders' checkers accept, equally for all of them.

## V3-046 (2026-10-29): fixes from the second code review (Gemini Argon), before the lock

What: the human had the current code reviewed again (`review/run5_code_for_review.md` with `episode.py` and
`render.py` added, plus `review/run5_changes_since_brief.md`); the reply is
`C:\Users\jsund\Downloads\run_5_second_code_review.md` (14 findings; earlier fixes confirmed or flagged). Each was
checked against the code. Not a bug: F2 (the Codex builder sets the error for a forbidden tool call in either phase,
so the run is voided). Fixed:
- F1: only the episode's own wall-clock limit ("rc=-9: timeout") is a design-induced failure; network and service
  timeouts are infrastructure (backoff, then halt).
- F5: template-first designs (SPEC, SEC_FLOW) had no read tools in the answer phase (the template gate applied to
  answering); they now get show_model like the other sectioned founders. Also affected runs 1-4.
- F6: an ANSWERS block in a code fence is read (it scored 0 before).
- F3: B is bounded by its length rule alone (a second 1,200-character cap contradicted it).
- F4: for a corrected key (fix_key) the leak answer is re-graded against the corrected key, acceptance and weights
  use that key, and the corrected-key grade must have succeeded.
- F7: the leak check fails if any key point tied for the highest weight is hit (not only the first of them).
- F8: "key events" or "each part links" are no longer taken as naming a section; the winner's answering text is
  checked with its fact-writing rules for names A+ must keep.
- F9: the recipe / compute split is reported on the sealed items only; if the final design is design 17 its whole
  lead is reported as compute.
- F10: the dashboard counts design-induced failures as 0 and keeps BLIND out of the live tests.
- F11: the judge calibration reads Opus's last grade object and checks the Claude allowance first.
- F12: a placeholder cannot contain "and" / "or" or digits, so "T1<T2 and P1>P2" is not one.
- F13: results by problem family (the base system's family) are reported; a restart clears a stale "waiting for";
  own-code workers waiting for Docker stop when the run stops; the build-rate check uses every item.
- F14: the winner's example must be rewritten for the new layout whatever the core; the new-layout example must
  differ from the old one and use the new layout's names.
Questions answered: all founders have max_turns 12, so E+ doubles to 24; by-family groups items by their base
system's family; the answering procedure (C) is shown in both phases, as in runs 1-4 (unchanged: knowing how it
will answer may shape the notes, and that is part of the design being tested).

Why: correctness before the lock.

Effect on results: nothing in run 5 has run (pilots only). F5 and F6 also affected earlier runs' scoring of SPEC and
SEC_FLOW (answer-phase tools) and of fenced answers; noted for the paper.

## V3-047 (2026-10-29): fixes from the third code review (Gemini Argon), before the lock

What: third review of the current code (`C:\Users\jsund\Downloads\run_5_code_review_round_3.md`): 9 findings, 7 of
the 14 V3-046 fixes confirmed correct and 7 incomplete (all covered below). Fixed:
- F2: a Docker outage during an episode was caught as a tool error and the run graded; the sandbox now records the
  outage, the MCP server writes it to its state file, and the Codex builder makes it an infrastructure error
  (backoff, then halt).
- F3: the judge calibration runs before stage 1 hands over and survives the Claude allowance stop (SystemExit), so a
  restart cannot skip it; a failed or timed-out Opus call while writing the + texts counts as a failed attempt.
- F4: founders are ranked on exact means (rounding to four places could create a false tie, broken by the
  hand-written mean).
- F5: OWN's form is classified on the notes after the last FINAL ANSWER marker.
- F6: a placeholder may contain only letters, spaces, "/" and "-" (comparisons such as "x<y, a>b" no longer match);
  unfilled Markdown table header rows are caught.
- F7: a voided run is not counted as "built" in the conditional accuracy.
- F8: a new leak answer clears the old corrected-key grade; a failed judge call in the leak re-check is retried
  later, never saved as a rejection.
- F9: a zero effect with zero spread gives t = 0 (not infinity); the mini run's stage 2 uses two base systems.
- F1: the rewritten example is checked whatever the core and whichever example it rewrites (not identical to the
  old, no empty example, and for a notes core it uses a name new in layout A+). Not changed: names the winner's own
  rules use are matched as words, so A+ keeps every key JSON's instruction names ("things: ...", "rules: ..."): the
  instruction really refers to them, and A+ still differs through depth, added keys or (Markdown) which sections are
  tables.
Questions answered: the ontology core has no placeholder check; no founder starts with a worked example; the MCP
server catches tool exceptions (hence the F2 route through its state file).

Why: correctness before the lock.

Effect on results: nothing in run 5 has run (pilots only).

## V3-048 (2026-10-29): fixes from the fourth code review (Gemini Argon), before the lock

What: fourth review (`C:\Users\jsund\Downloads\run_5_code_review_round_4.md`, verdict "not ready to lock"): 6
findings; 5 of the 8 V3-047 fixes confirmed, 3 incomplete (covered below). Fixed:
- F1 (critical, confirmed): the schema gene reaches the builder only through get_template, which only template-first
  designs offer. For the five one-shot calculator and own-code founders (G1C3, A5, A2, A3, G1C1), factor A as a schema
  change would have been invisible: 16 A+ designs identical at run time to 16 A- designs. Factor A is now, for every
  core without layout parameters, a paragraph on how to organise the notes or model (grouping, order, naming) added
  to the writing guidance, which the builder always sees. A general guard: the 32 designs must render to 32 distinct
  guides / tool sets / limits, or stage 2 does not start; tested for all ten founder kinds. Also: in runs 1-4 the
  schema gene had no effect on one-shot designs (noted for the paper).
- F2: a Docker outage is an infrastructure failure on every path: checked before the allowlist check (which had
  turned it into a config error and stopped the retries), Docker failing to start a container (exit 125) counts as
  an outage, and a retry never reads the previous attempt's tool-server state.
- F3: the Claude allowance stop pauses the run before the judge calibration (instead of skipping it) and before Opus
  writes the + texts; one failed Opus grade in the calibration loses that grade only.
- F4: a rewrite counts as accepted only if it was submitted (a rewrite that never started no longer passes); the
  rewrite feedback and the leak re-check use the corrected key where there is one.
- F5: exact means are compared to 12 decimals, so float noise cannot break an exact tie (the hand-written mean does);
  groups and Spearman use exact means; a zero effect is zero below 1e-12.
- F6: the Codex usage check counts the call first and survives a timeout.
Question answered: prose designs answer from their whole build reply (reasoning and notes), as in runs 1-4, the same
for every prose design.

Why: correctness before the lock.

Effect on results: nothing in run 5 has run (pilots only). Factor A for a calculator or own-code winner is a layout
paragraph, not a schema change.

## V3-049 (2026-10-29): fixes from the fifth code review (Gemini Argon), before the lock

What: fifth review (`C:\Users\jsund\Downloads\run_5_code_review_round_5.md`, verdict "not ready to lock"): 5
findings; 4 of the 6 V3-048 fixes confirmed, 2 incomplete. Fixed:
- F1 (critical, my error in V3-048): an inline comment placed mid-line in analysis5.compare commented out the 95%
  range of every design (stage 1, hand-written, baseline, final check, dashboard). Restored, with a test that every
  design summary has its range.
- F2: factor E (the thinking budget) changed only a hidden cap on tool calls; Luna was never told, and for the prose
  founders (A0P, OWN), which have no tools, it changed nothing. Now every tool-using design's guide states its tool
  budget, and a prose design with the doubled budget gets one revision turn after its notes (announced in its guide;
  run by v3.episode._two_phase). The 32-distinct-designs guard counts only what Luna sees (guide and tools), not the
  hidden cap; it passes for all ten founder kinds tested.
- F3: least-squares noise cannot pick the final design: coefficients below 1e-12 are zero, fitted scores are rounded
  to 12 places, and a tie keeps the lower design number (design 1, the winner).
- F4: a winner without a worked example (all founders) cannot have a rewritten winner example; build never uses one
  for such a winner.
- F5: a Codex reconnect message followed by a completed turn is not an error.
Also: the Codex usage counter is put back when the guard pauses, so the next call checks again (V3-048 F6 had
skipped the check after a pause); the sandbox cleanup test checks for stopped containers left behind only (running
ones may belong to other jobs).

Why: correctness before the lock.

Effect on results: nothing in run 5 has run (pilots only). Every tool-using founder's guide now states its tool budget
(the same line for all, 12 in stage 1).

## V3-050 (2026-10-29): hardening from the sixth code review (Gemini Argon); verdict "ready to lock"

What: sixth review (`C:\Users\jsund\Downloads\run_5_code_review_6.md`): verdict ready to lock; all V3-049 fixes and the
two extra patches confirmed, no earlier fix broken; one medium and three low hardening points, applied before the lock
(tests written first and seen to fail on the old code, then passing; diff read line by line):
- F1 (medium): Docker hanging during kill / rm no longer escapes as a tool error, and a failed container is checked
  for an outage even after a timeout (a timeout because Docker died is an outage).
- F2: an Opus reply with the wrong shape (a factor as a string or list, null headers) is a failed attempt, not a crash.
- F3: a corrected key must have a summary and key points with id and text; it keeps the original wrong answers when
  it gives none (used everywhere a corrected key is used).
- F4: the prose revision note speaks of the first draft only.
- Q1: "built" has one definition (usable and executable, as v3.episode.buildable).

Why: the human asked for the code to be reviewed until ready, and for every fix to be double-checked.

Effect on results: nothing in run 5 has run (pilots only).


## V3-051 (2026-10-29): hand-written item text repaired; tool server reads UTF-8; item labels from the system record

What:
- The human's spot-check page showed garbled characters ("â€“" for "–", "mÂ³" for "m³") in 412 of the 500 hand-written
  items (and 3 of 3 pilot items). Cause: `v3/builders/mcp_server.serve` read stdin with the Windows code page
  (cp1252). The Claude CLI sends raw UTF-8, so every non-ASCII character the item writers wrote through `write_file`
  arrived mangled. Codex escapes non-ASCII in its messages, so Luna's notes were never affected (checked: 117 of 121
  format-pilot notes contain non-ASCII characters, none garbled).
- Items: `v3.hand5.factory.final_item` now undoes the mangling (`unmangle`: encode cp1252 with surrogateescape,
  decode UTF-8; that decoding is lossless). Checked: same 500 ids; every changed string re-mangles to the old one
  exactly (context 415, question 26, answer key 22, construction notes 4, system title 2). Old files archived in
  `items/hand5/archive/`.
- Items: kind, domain and messiness come from the system record (the writer check holds it to the plan); 9 items
  had the task type written as their kind. Only that label changed.
- Tool server: `serve` reconfigures stdin to UTF-8. A test feeds raw UTF-8 through a cp1252 stdin; a real piped
  subprocess was checked before (mangled) and after (intact).
- Stage 2 prep brief (`v3/plus5.brief`): the mini run (2026-10-29, `runs/logs/run5mini_v3.log`) completed end to
  end, but its Opus needed all three attempts: attempt 1 named a header `OPEN_QUESTIONS`, which the schema refuses
  (the brief said only "upper case"). The brief now states the allowed characters (headers: upper-case letters,
  digits, single spaces and & / -, no underscores; JSON keys: lower-case letters, digits and _). Approved by the
  human ("do the one line fix and retest"). Retest with real Opus on all 8 founders with core-parameter layouts: no
  invalid names; 7 of 8 passed the fit check on the first attempt; the eighth (STATES) was an unreadable-reply
  failure (retried by the engine), and two more STATES replies both passed.

Why: the human found it on the spot-check page; the human approved the server fix ("yes").

Effect on results: none on run 5 scoring rules. Luna's runs are unchanged (her client already sent ASCII). The
item checks (validator, leak, adjudicator) ran on the mangled text; the leak check never sees the document and the
validator could only find clean text easier, so items that passed remain valid. The human spot-checked the repaired
text ("it is good"). Applied before the run 5 lock; nothing in run 5 has run (pilots only).

## V3-052 (2026-10-29): run 5 locked and launched

What: after V3-051, the full suite (400 tests) passed, the mini run completed end to end and the item-range check
found no overlaps. `config/v3.json` sets `"run": 5`; `python -m v3.prereg lock --reason "V3-052: run 5"`, then
verify; `scripts/run_v3_loop.cmd` now starts `python -m v3.run5 loop` (log `runs/run5_loop.log`); `state/PAUSE`,
which guarded the archived run 4 loop, is removed so the run 5 loop can start (run 4's loop is no longer started by
anything). Dashboard: `python -m v3.dashboard5.server5 --port 8766 --mode real`.

Why: the human's go ("if all your code and scripts check out, we go ahead"; "no go ahead").

Effect on results: this is the start of run 5.

## V3-053 (2026-10-29): dashboard server refresh (display only), re-lock

What: the human asked for the dashboard to refresh every second without risk to the run. `v3/dashboard5/server5.py`
rebuilds the feed at most once a second, and a request that arrives during a rebuild gets the last result at once
(tested: 0.0 s wait during a 1.5 s rebuild; a failing rebuild serves the last result and frees the lock); the page
asks again only after each answer. The prereg lock covers `v3/dashboard5/*.py` (only a folder named `dashboard` is
excluded), so the lock is renewed: `python -m v3.prereg lock --reason "V3-053: dashboard 5 server refresh"`.

Effect on results: none (the dashboard only reads records). The run 5 loop kept running; it checks the lock only
when it starts.

## V3-054 (2026-10-29): dashboards are never locked

What: the human asked why the dashboard was in the lock. The lock was meant to leave out the dashboard (it only
reads records), but `v3.prereg.files` skipped only a folder named exactly `dashboard`, so `v3/dashboard5/*.py` was
locked by accident (this forced V3-053's re-lock). Now any folder whose name starts with "dashboard" is left out.
Test: `test_prereg_lock_leaves_out_every_dashboard` (fails on the old rule). Re-locked.

Effect on results: none (no experiment code, item or rule changes; the locked set only loses the dashboard files).
The run 5 loop kept running.

## V3-055 (2026-10-29): the state save waits out a brief file lock

What: at 16:54 on 29 Oct, the moment all 15,680 stage 1 runs were finished, the run 5 loop stopped while saving the
hand-over to stage 2 prep: Windows refused to replace `state/run5/state.json` (`PermissionError: [WinError 5] Access
is denied`), most likely because the dashboard (which reads it every second) or OneDrive had the file open at that
instant. `save_state` in `v3/run5.py` now retries the replace for up to about 10 s (20 tries, 0.5 s apart) before
giving up; a lasting lock still raises. Tests: `test_save_state_retries_when_the_file_is_briefly_locked` (fails on
the old code) and `test_save_state_still_fails_on_a_lasting_lock`. Approved by the human ("do the fix then restart
it"). Re-locked; the loop is restarted through the runner.

State at the stop: every stage 1 run and score was already recorded; `state.json` still said stage 1 and the
unsaved hand-over (winner FACTS, runner-up A3, tied at the top; hand-written top ONT_LOOSE) was left in
`state.tmp`. On restart, stage 1 finds every run done, so nothing is re-run; it recomputes the same stage 1 analysis
and repeats the 100-answer Opus judge calibration (reported, never gating), then hands over to stage 2 prep.

Effect on results: none (only how a file is saved; no rule, item, score or analysis changes).

## V3-056 (2026-10-30): the stage 2 "+" texts are written from the stage 1 evidence

What: Opus wrote and froze the five "+" texts for FACTS (`plus.json`, hash 2dab40bb9fc2e92b) from first principles,
as pre-registered, seeing no stage 1 result. The human objected that the changes "should be based on data". After a
Gemini Argon review, the human approved a firewalled middle path: Opus writes new texts with a short summary of the
winner's stage 1 evidence on the generated items, and nothing that could bias stage 2 or the final check.
- Shown (`v3/plus_evidence5.py`, saved in `state/run5/plus_evidence.json`): the winner's mean on the 480 generated
  items (0.657); its mean by k (0.772 / 0.713 / 0.631 / 0.513); notes-check failures (55 of 480, 11.5%) by the
  checker's fixed problem category (more than one fact in one element 41, element longer than 300 characters 25,
  line not an element 1; quoted names dropped, never notes text) and by k (0, 0, 3 and 52 of 120); that the builder
  is not told about checker problems during the build; build tool calls by k (mean 1.1 at k = 1 to 3.0 at k = 8,
  never more than 6); output tokens; the tool-call budget (12). The by-k form of the tool calls was chosen after an
  Argon review warned that calls split by pass/fail alone could be misread as "more calls cause failures". The brief
  asks for parts that a small model can follow without tripping the checker and that help at k = 8.
- Factor E (fixed by code: the tool-call budget 12 -> 24, shown to the builder) is unchanged. No stage 1 build used
  more than 6 calls, so E+ may well have no effect; the pre-registered design keeps it, and a null E is reported.
- Never shown: anything from the 500 hand-written items (the final check reuses them; showing them would spoil it);
  scores by problem family (they invite domain tricks rather than a better general format); the pass-only score
  (0.676); any question, question kind, item, answer key, notes text or tool trace. The module reads only the
  winner's records on the generated items and only the fields in `SAFE_FIELDS`.
- Code: `plus()` in `v3/run5.py` archives unapproved texts written without the evidence to `plus_blind_<n>.json`
  (not a rejection; the two-rejection limit is unchanged), writes the evidence file, passes it into Opus's brief
  (`v3/plus5.py`, new EVIDENCE block) and records its hash in `plus.json` (`evidence_hash`). Approval works as before.
- Tests: `test_plus_evidence_firewall` (a guard fails on any read of a hand-written record or an unlisted field;
  planted strings in checker messages, notes, answers and a family field must not appear),
  `test_plus_evidence_categories`, `test_plus_brief_carries_the_evidence` and
  `test_blind_plus_texts_are_archived_and_rewritten` (the last two fail on the old code).

Caveat: stage 2 reuses the 480 generated items of stage 1, and the evidence comes from them, so the stage 2 effects
may be slightly optimistic. The evidence is aggregate (eleven numbers and three checker categories), and the sealed
720-item final check stays the unbiased estimate. The paper reports this as a deviation.

Effect on results: changes the stage 2 "+" texts (factors A to D; E is fixed by code). No item, score, rule or
analysis changes. Decided before any stage 2 run.

Applied 2026-10-29 22:14 (re-locked, paused, restarted through the runner). Opus wrote new texts in 43 s (hash
9a641687b2cd30fa, evidence hash 4dbc2e4551542717); they passed the fit check and waited for approval. They were not
approved: see V3-057.

## V3-057 (2026-10-29): the evidence brief states the build tools correctly

What: reviewing V3-056's texts, the human asked how the save tool is described to the builder. Checking it, Claude
found that one sentence of the V3-056 brief was false: "The builder is not told about checker problems during the
build." In fact `v3.genome.render._core_call` appends the checker's result to every save of a v3 core, and FACTS's
errors style is first_only, so after each save the builder sees the first problem with its line number and how many
more remain. Claude had read only the inner save message (`v3.cores.sectioned._report`) and missed that layer. The
V3-056 texts also asked the builder to "save in several passes, one subsystem at a time", although every save
replaces the whole notes (the tool says so, but Opus never saw the tool descriptions).
- The brief now states how the build tools work, read from the rendered design, never assumed
  (`plus_evidence5.tool_facts`): "Each save replaces the whole notes; there is no append. So a part that asks the
  builder to save in several passes makes it rewrite everything written so far on every save." and "After each save,
  the builder is shown the first checker problem (with its line number) and the count of how many more problems
  remain. So fixing reported problems one save at a time uses up tool calls from the budget." This describes the
  design, not stage 1 results, so the firewall is unchanged.
- `plus()` now archives unapproved texts whose `evidence_hash` differs from the current brief's as
  `plus_superseded_<n>.json` (not a rejection), then Opus writes new ones. The V3-056 texts are archived this way.
- A rejection (PLUS_REJECT) still passes no reason to Opus: it is a re-roll of the same brief, which is why this was
  a locked change and not a rejection.
- Tests: `test_plus_evidence_tool_facts_match_what_the_builder_sees` (checks the facts against a real save on the
  FACTS design) and `test_plus_texts_from_an_earlier_evidence_brief_are_archived`; both fail on the V3-056 code.
Approved by the human after an Argon review.

Effect on results: changes the stage 2 "+" texts again (factors A to D). No item, score, rule or analysis changes.
Decided before any stage 2 run.

## V3-058 (2026-10-30): the gated retention rule; the opened sealed set retired; stages 3 to 5; a fresh sealed set

What happened: stage 2 ended at 11:18 on 2026-10-30 (32 designs x 480 items, none dropped). No new part helped: A
-0.017, B -0.035, C -0.043 (all Holm-significant, harmful), D -0.003, E +0.003 (unclear); the winner unchanged (D01 =
FACTS) had the highest raw mean, 0.660. The locked rule (the argmax of the fitted mains + pairs model) picked D18
(A+, E+; raw mean 0.651, fitted 0.6565 against D01's 0.6484) through the negative A:B and A:C interactions, which
made A+ look good at B-, C-. The final check started on the sealed set, opened at 11:18. The human paused the run at
12:23 ("why would the trust pick a worse combination"); the loop exited at 12:25:04. 1,435 sealed runs had been
recorded (about 239 each for FACTS, A3, A0P, ONT_LOOSE, D18 and D17). No sealed score was viewed by anyone.

Decided by the human, after two reviews with Gemini Argon (design: `docs/RUN5_STAGES_DESIGN.md`, review log in its
section 14): FACTS vs FACTS on the sealed set would waste the check, and finishing a check on a worse design makes no
sense; the run's rules are broken either way, so run 5 continues with further improvement stages and the trust score
comes at the end, on a fresh sealed set.

The change:
1. Gated baseline retention rule (`v3.analysis5.retention`; design section 3), for stage 2 now and every
   confirmation: mains surviving Holm with a positive estimate are kept; a two-way interaction only if it survives
   Holm and both its mains are kept (marginality); a factor is never set to + solely through a negative interaction;
   d* = the argmax of the kept terms; d* replaces the current best only if its direct paired contrast with it across
   base systems has its 95% t range above zero. On stage 2 (Holm family: the nine pre-registered terms) nothing is
   kept: the verdict is D01 = FACTS (checked on the real records before the change; the old rule on the same data
   still gives D18). The fitted pick is kept in the state as `final_design_fitted`; the stage 2 analysis is
   unchanged, the verdict is a new part `stage2_retention` of `runs/run5/analysis.json`.
2. The opened sealed set is retired (`v3.run5_sets.RETIRED`): its records stay on disk unchanged; a manifest of
   every file of every retired run folder with its SHA-256 (1,435 record.json, 7,184 files, manifest hash
   57069cf842ec925d155e94c71fabd58300a75f1846cc6bea94d867af04ebbd4c) is in `state/run5/retired_V3_058.json`
   (`scripts/retire_sealed_v3_058.py --verify` checks it). `v3.analysis5.matrix`, `v3.run5.record` (run 5's own
   folder) and `v3.plus_evidence5.summary` raise `Retired` on any of its items. The dashboard shows its record count
   only. Its partial means may be disclosed in an appendix after the final verdict, never before.
3. Stages 3 to 5 (`v3/stages5.py`, `v3/run5.py`; design sections 4-7, 9-10), at most three:
   - five slots of the current best: 1 layout (`core.params`), 2 fact rules (`guidance.text`, first block), 3 build
     check (a separate second block of `guidance.text`; empty in FACTS, so FACTS's guidance is unchanged), 4
     answering (`answering.text`), 5 worked example (`examples.items`, always in the design's own layout and fact
     rules, indexed by the slot 1, 2 and 5 levels; written by Opus; must pass the design's notes checker). The old
     factor E (turn budget) is no longer a slot.
   - write: Claude Opus writes 4 candidates per slot, one per archetype (1 subtract - for an empty slot the smallest
     useful addition; 2 explicit structure; 3 verify; 4 targeted tweak), one fresh session per slot with no tools,
     from the current best's genome and the firewalled evidence (`plus_evidence5.stage_brief`: its score overall
     and by k, checker failures by category and k, tool calls, tokens, tool behaviour, on the previous stage's
     confirmation items; earlier candidates with their measured effects, full text for the advancing and worst ones;
     the B x D lesson; never items, questions, answer keys, hand-written results, family scores or sealed records).
     Mechanical checks: valid and buildable, only its own slot changes, slots 2-4 name no section header of any
     layout in play, slot 3 quotes or defines no syntax, lengths (fact rules from half the current length to 2,500
     characters; build check and answering at most 2,500; example at most 4,000 - raised by the human 2026-10-30
     from the stage 2 limits; every candidate's length is recorded and shown), diversity (archetypes 1-3 differ
     pairwise by at least 25% token edit distance; archetype 4 changes at least 10% and adds nothing archetypes 1-3
     also add; header lines do not count as content), all 21 screen designs render differently. A failing slot goes
     back to Opus with the reasons, up to 3 attempts. Then frozen (hash) and approved by the human
     (`state/run5/stage<n>/APPROVED`); REJECT sends named slots back for a listed reason, at most twice.
   - screen: the current best + the 20 candidates (one slot changed each) x 320 fresh items (80 bases; option S3 of
     the second Argon review; measured SE of a paired difference about 0.0153), in `runs/run5_s<n>a`. A slot's best
     candidate advances if its paired lead (averaged within base systems) is at least +0.015; no k = 8 gate (it is
     reported); at most 4 slots, the largest leads (human 2026-10-30). Futility stop if none advances.
   - examples: Opus writes the worked examples the confirmation still needs (the same content in another layout or
     fact rules; checker-clean); the human approves them (EXAMPLES_APPROVED) only when any were written, since every
     other confirmation text was approved before the screen.
   - confirm: the 2^m factorial (m <= 4) over the advancing slots x 480 fresh items (120 bases), in
     `runs/run5_s<n>b`; D01 = the current best, run fresh; Holm over the m mains + m(m-1)/2 pairs; the retention
     rule decides the new current best. Convergence stop if it stays or the gain is under +0.005; the stages also
     stop after stage 5.
4. Fresh item sets (design section 12), chosen once and frozen in `items/run5_stages.json` (sha256
   b4a4e0bf98ac3b47b649109b534ce5862f0cb36675141fbf7cc2803410b507b5; hashed into the lock): s3/s4/s5 screen (80 bases
   each) and confirmation (120 each) in the validation split, and sealed2 (180 bases) in the train split; free
   500-seed windows (no folder K<k>x<base> or W<seed> anywhere in rpp-v3, no other run 5 set, none of
   `v3.run5_sets.EXCLUDED_RANGES`: calibration 1,900,000-2,000,000, the hand-of-God test 850,000-907,000, the seeds
   v2 selected its designs on - micro panels 0-5,000 and 1,000,000-1,005,000, meso 500,000-560,000 and
   1,500,000-1,560,000 - and run 3's configured confirmation range 20,000-80,500), round robin over the 7 problem
   families (balanced within one base per family in every set), cut in a fixed order. A first allocation, made the
   same evening before any run or lock, put sealed2 at seed 0 and s3_screen at 1,000,039, inside v2's micro panels
   (found by the full test suite: old tests hit the new seal); it is archived unused as
   `items/archive/run5_stages_superseded_1.json` and the sets were chosen again with the v2 ranges excluded. `scripts/check_item_ranges.py --run5`: no overlap
   (it now treats a W item as its single seed and a set's own runs as no overlap). sealed2 is sealed until the final
   check (`v3.run5_sets.SEALED`).
5. The final check (`v3.run5.final2`, `v3.analysis5.final_stages`; design section 8), after the last stage, on
   sealed2: the final design, FACTS, A0P, ONT_LOOSE (hand-written top) and A3 (runner-up); fixed sequence at alpha
   0.05, each counted only if every earlier one passed (95% t range across base systems above zero): 1 vs A0P, 2 vs
   ONT_LOOSE, 3 vs A3, 4 vs FACTS (only if they differ). Also FACTS vs A0P and scores by k. The final design on the
   500 hand-written items (the others' stage 1 records reused), vs FACTS as a paired test across the items (df 499).
   The old final check (`run5.final`, and its recipe / compute split) is removed; the state "final" now hands over
   to the stage 2 verdict.
6. Dashboard (not locked): stages from the state, a tile and a tab per stage (candidates with their lengths, screen,
   confirmation), the stage 2 interaction panel renamed "Interaction terms (synergy / antagonism)" with each pair's
   design score and its difference from D01, the new final check, the retired set's record count only.

Tests: `tests/test_stages5.py` (the retention rule keeps D01 on the stage 2 pattern where the old rule picks an A+
design - it fails on the old code, which has no such rule; real gains, marginality, the direct gate; screen gate and
cap; the fixed sequence; slot isolation and checks; the example cube; the writer's retries; the brief's contents; the
retired set refused by the analysis, the records reader and the evidence; the allocation disjoint and balanced),
the mini run end to end through stage 3 and the new final check, and the dashboard feed on it (`tests/test_run5.py`).

Effect on results: stage 2's results are unchanged; its verdict is FACTS, not D18. The sealed set opened at 11:18 is
not used for any result. The stages add up to 47,300 runs at most (3 x (6,720 + 7,680) + 4,100), about 36 h. The
stage 3-5 results use fresh items each stage, so they are not biased by reuse; the fresh sealed set gives the
unbiased trust score. Disclosed in the paper: the rule change after stage 2 and before any sealed score was seen,
and the retired set.
