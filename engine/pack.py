"""The problem pack: everything the engine needs to know about one problem, and nothing else.

The engine never looks inside a candidate, an item or a record. It asks the pack to run a candidate on an item and get
a score, to split a candidate into its parts, to write new versions of a part (from a firewalled brief), and to put
parts back together. Subclass Pack and fill in the methods marked "required"; packs/toy/pack.py is a complete example
that runs in a second, and docs/WRITING_A_PACK.md explains each method.

Vocabulary
  candidate  a JSON-serialisable dict with a stable "id" (equal contents must give an equal id): a prompt, a note
             format, an agent's instructions, a configuration
  item       a test case, named by a string id; the pack loads it, runs it and scores it
  part       one slot of a candidate (for example layout, rules, answering procedure, examples, budget); the engine
             changes parts one at a time and in every combination
  alt        a new version of one part: {"value": ..., "rationale": "...", plus anything the pack needs}
  record     what one run returned: {"score": 0..1} or {"error": "...", "failure": "design" | "infra" | "config"}
"""


class Pack:
    name = "unnamed"
    description = ""

    # ------------------------------------------------------------------ required: candidates, items and scoring
    def founders(self):
        """{label: candidate}: the candidates that sit the first exam, the current setup (the control) among them."""
        raise NotImplementedError

    def item_sets(self):
        """{set name: [item id, ...]}. The plan says which set each step uses; the set it names as sealed is never run
        before the final check."""
        raise NotImplementedError

    def run(self, candidate, item, ctx):
        """Run one candidate on one item and score it. ctx: {"stage", "folder", "dir"} (dir: a folder the pack may
        write its own detailed record into). Return {"score": float in [0, 1], ...} or, when the run failed,
        {"error": str, "failure": kind}:
          "design"  caused by the candidate meeting the item (it broke a rule, ran out of time): scored 0 after the
                    retries, never dropped
          "infra"   anything else (connection, service, grader): retried with a backoff, then the run stops for a
                    person; never scored
          "config"  the candidate cannot run at all: the run stops at once
        Raise an exception named Paused to stop the run cleanly (for example an allowance guard)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ required for steps 4 and 5
    def slots(self, kind):
        """The part names of a candidate, in a fixed order. kind: "takeapart" (step 4) or "buildout" (step 5)."""
        raise NotImplementedError

    def parts(self, candidate, kind):
        """{slot: value}: the candidate's current version of every slot."""
        raise NotImplementedError

    def write(self, kind, candidate, parts, brief, slots, per_slot):
        """The designer: new versions of the given slots. brief is the engine's firewalled summary (scores only, never
        an item, question, answer key or sealed result); a pack must not add item content to it. Return
        ({slot: [alt, ...]}, extras) with per_slot alts per slot (or the pack's own fixed number), each already checked
        by the pack's mechanical rules. extras: anything compose() will need later (stored and frozen with the alts)."""
        raise NotImplementedError

    def compose(self, base, parts, changes, kind, extras):
        """A candidate: `base` with its slots set from parts, then the slots in changes {slot: alt} replaced by those
        alts. No changes must give back a candidate with base's id."""
        raise NotImplementedError

    # ------------------------------------------------------------------ optional
    control = None                     # the founder label of the current setup (or set it in the plan)

    def cluster_of(self, item):
        """Items that share a source (for example one system asked at four sizes) form a cluster; ranges and tests
        resample clusters, not items. Default: every item is its own cluster."""
        return item

    def level_of(self, item):
        """A difficulty level for the "by level" tables (for example k = 1, 2, 4, 8), or None."""
        return None

    def failure_type(self, record):
        """A short label for why a run lost points (for the "why it fails" tile), or None."""
        return None

    def evidence(self, candidate, records):
        """Extra evidence for the designer's brief, computed from this candidate's records: aggregates only (counts,
        means, lengths), never an item's text, a question, an answer or a key. Return {name: value} or {}."""
        return {}

    def prepare(self, kind, candidate, parts, advancing, extras):
        """Called before a confirmation with the alts that passed the screen {slot: alt}: write anything the
        combinations still need (for example a worked example in a new layout). Return the updated extras, and
        whether a person must approve the additions (True/False)."""
        return extras, False

    def open_sealed(self, set_name, reason):
        """Called once, when the final check opens the sealed set."""

    def retired_items(self):
        """Item ids no analysis may ever read (for example a sealed set that was opened and then withdrawn)."""
        return frozenset()

    def report_extras(self, stage, scores, items):
        """Pack-specific tables to add to a stage's analysis (reported, never used to decide)."""
        return {}


class Paused(RuntimeError):
    """Raised by a pack (or the engine) to stop a run cleanly; running jobs finish, nothing is lost."""
