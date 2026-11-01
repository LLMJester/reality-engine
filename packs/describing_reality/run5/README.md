# The code that ran run 5

This folder is the code that ran *Describing Reality for AI* (run 5 of the Discovery Program), copied unchanged.
`MANIFEST.json` lists every file with its SHA-256 and the commits it came from. The results, the paper and the rules
the run was locked to are in [describing-reality-for-ai](https://github.com/LLMJester/describing-reality-for-ai).

The pack in the folder above (`../pack.py`) connects this code to the engine. Read it here to see exactly how the case
study was built: what a design is, how the notes are written and checked, how the answers are graded, and what the
designer AI is asked and how its versions are checked.

| Folder | What it holds |
| --- | --- |
| `rpp-v3/v3/` | The problem: `genome/` (what a design is and how it becomes instructions; the 15 founders), `episode.py` (write the notes without the questions, then answer from the notes alone), `grading/`, `plus5.py` and `stages5.py` (the designer's prompts and checks), `run5_sets.py` (the item sets), `hand5/` (the factory for realistic hand-written documents), `sandbox.py` (Docker, for notes written as code), `run5.py` and `analysis5.py` (run 5's own loop and statistics, which the engine now replaces), `dashboard5/` |
| `rpp-v3/tests/` | The test suite (models are faked) |
| `rpp-v3/sandbox/` | The Docker image for model-written code |
| `rpp-core/` | The earlier framework's parts that run 5 uses: the note formats (`tournament/arms/`), the model-call layer, the judge and the prompts |

Not here: the item generator (`rpp-core/discovery/world`), any items, answer keys or records. With the generator,
anyone could rebuild the answer keys of the sealed sets, so it stays private. Code that makes or loads items needs it.
