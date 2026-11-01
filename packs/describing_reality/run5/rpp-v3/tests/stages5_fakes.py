"""A fake Opus for the run 5 stages (V3-058): answers the stage 2 + brief, the slot briefs and the example briefs with
replies that pass the mechanical checks for FACTS. Used by tests/test_run5.py and tests/test_stages5.py."""
import json

EX_SMALL = ("## THINGS\n- Farm.T1: water tank\n## VALUES\n- Farm.T1.level: 40% at start\n## RULES\n"
            "- if Farm.T1.level > 90% then Farm.P1 stops\n## LINKS\n- Farm.P1 -> Farm.T1: fills\n## TIMING\n"
            "- Farm.P1: starts 06:00 daily")
EX_IDS = ("## THINGS\n- id P1: Mill pump\n- id T1: Mill tank\n- id V1: Mill valve\n## VALUES\n- P1.flow: 20 L/min when "
          "on\n- T1.capacity: 800 L\n## RULES\n- if T1 full then P1 off\n## LINKS\n- P1 -> T1: fills\n- T1 -> V1: "
          "drains through\n## TIMING\n- P1: runs from 08:00 to 17:00")
EX_CHECK = ("## THINGS\n- Lab.F1: freezer\n- Lab.A1: alarm\n## VALUES\n- Lab.F1.setpoint: -20 C\n## RULES\n"
            "- if Lab.F1.temp > -15 C then Lab.A1 sounds\n## LINKS\n- Lab.F1 -> Lab.A1: signals\n## TIMING\n"
            "- Lab.A1: delay 5 min after the trigger")
EX_TWO = ("## THINGS\n- Dock.C1: crane\n- Yard.K1: truck\n## VALUES\n- Dock.C1.lift: 30 t maximum\n- Yard.K1.load: unknown"
          "\n## RULES\n- if Dock.C1.wind > 50 km/h then Dock.C1 stops\n## LINKS\n- Dock.C1 -> Yard.K1: loads\n"
          "## TIMING\n- Yard.K1: leaves every 2 h")

SLOT_REPLIES = {
    1: [{"headers": ["THINGS", "VALUES", "LINKS"], "tags": False, "granularity": "fine"},
        {"headers": ["ENTITIES", "EDGES", "STATES", "RULES", "TIMING", "UNKNOWNS"], "tags": False,
         "granularity": "fine"},
        {"headers": ["PARTS", "QUANTITIES", "CAUSES", "CHECKLIST"], "tags": True, "granularity": "coarse"},
        {"headers": ["THINGS", "VALUES", "RULES", "LINKS", "TIMING", "OPEN POINTS", "IDS"], "tags": False,
         "granularity": "fine"}],
    2: ["Write each fact as one short bullet: the name, then its value or relation. No prose and no repetition; one "
        "fact per bullet, under the section it belongs to.",
        "Give every part a short unique identifier and use it in every bullet that mentions the part. Write each link "
        "between parts as source identifier, relation, target identifier. Write each state change with its trigger "
        "and its time. One fact per bullet.",
        "After each bullet, ask whether it holds exactly one fact and whether every number keeps its unit. Prefer many "
        "short bullets over one long bullet. Never merge two systems in one bullet; give names their system prefix.",
        None],                                            # 4: the current rules plus one sentence (set below)
    3: ["Save once, then read the checker report and fix only what it names.",
        "Before saving, list every system and every part the description names, and make sure each one appears in the "
        "notes together with its connections.",
        "After saving, read the notes back against the description paragraph by paragraph and add any missing fact "
        "before you finish.",
        "Before you finish, count the systems in the description and check that each system has notes about when "
        "things happen."],
    4: ["Answer from the notes only; if they do not say, answer unknown.",
        "1. Find the parts the question names. 2. Follow their connections one step at a time. 3. Read the values "
        "with units. 4. Answer.",
        "Find every relevant fact, compute the answer, then check it against each fact you used before you give it.",
        "When the question names several systems, answer for each system separately, then combine the results."],
    5: [EX_SMALL, EX_IDS, EX_CHECK, EX_TWO],
}
TWEAK = " Prefix every name with its system when the description covers several systems."


def slot_of(brief):
    for slot, key in ((1, "the LAYOUT of the notes"), (2, "the FACT-WRITING RULES"), (3, "the BUILD CHECK"),
                      (4, "the ANSWERING PROCEDURE"), (5, "the WORKED EXAMPLE shown")):
        if key in brief:
            return slot
    return None


def current_rules(brief):
    start = brief.index('(currently: "') + len('(currently: "')
    return brief[start:brief.index('"). Give a complete replacement', start)]


class FakeOpus:
    """call(system, user) -> reply text. plus_reply answers the stage 2 + brief; calls are logged."""

    def __init__(self, plus_reply, bad_slots=()):
        self.plus_reply, self.calls, self.bad_slots = plus_reply, [], set(bad_slots)

    def __call__(self, system, user):
        self.calls.append(user)
        if "Rewrite the worked example" in user:
            src = user.split("THE EXAMPLE TO REWRITE\n", 1)[1].split("\n\nIt must pass", 1)[0]
            heads = [ln[6:].rstrip("`") for ln in user.split("\n") if ln.startswith("- `## ")]
            facts = [ln for ln in src.splitlines() if ln.startswith("- ")]
            out, per = [], max(1, len(facts) // len(heads))
            for j, h in enumerate(heads):
                chunk = facts[j * per:(j + 1) * per] if j < len(heads) - 1 else facts[j * per:]
                out.append(f"## {h}\n" + "\n".join(chunk or ["- nothing stated"]))
            return json.dumps({"example": "\n".join(out)})
        slot = slot_of(user)
        if slot is None:
            return self.plus_reply
        if slot in self.bad_slots:
            return json.dumps({"candidates": [{"archetype": 1, "value": "x", "rationale": "r"}]})
        vals = list(SLOT_REPLIES[slot])
        if slot == 2:
            vals[3] = current_rules(user) + TWEAK
        return json.dumps({"candidates": [{"archetype": a, "value": v, "rationale": f"approach {a}"}
                                          for a, v in enumerate(vals, 1)]})
