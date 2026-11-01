"""The balanced plan for the hand-written-style items (code only, deterministic).

180 systems are built (540 items) so that 500 survive the checks; the final 500 are the first accepted items in plan
order. Two extra pilot systems (6 items) are for the mini run only and are never among the 500.
- domain: 18 domains, ten systems each, in rotation
- kind: continuous / discrete / stochastic in rotation (60 each), shifted every round of 18 so each domain gets a mix
- messiness: adversarial / realistic alternating (90 each), shifted the same way
- task types: an adversarial system always has a completeness question (its deliberate omissions are the point);
  the other slots cycle through prediction, diagnosis, intervention, assumptions, consistency, so all six types
  appear 90 times in the 540 slots.
"""
DOMAINS = ("chemical process plant", "electricity grid", "water utility", "hospital emergency department",
           "warehouse and distribution", "urban rail operations", "airport ground operations",
           "cloud software service operations", "manufacturing supply chain", "irrigated agriculture",
           "fisheries and ecology", "payments processing operations", "assembly line manufacturing",
           "telecom network", "emergency dispatch", "building heating and cooling", "customer contact centre",
           "underground mining operations")
KINDS = ("continuous", "discrete", "stochastic")
OTHERS = ("prediction", "diagnosis", "intervention", "assumptions", "consistency")
N_SYSTEMS = 180
N_ITEMS = 500
PILOT = ("HP1", "HP2")
# V3-044 (Argon review, human 2026-10-29): diagnosis questions fail the leak check far more often than the others,
# and the 500 are chosen by quota per question type, so 60 top-up systems each carry a diagnosis question.
N_TOPUP = 60


def _types(adversarial, ptr):
    out = ["completeness"] if adversarial else []
    while len(out) < 3:
        t = OTHERS[ptr % len(OTHERS)]
        ptr += 1
        if t not in out:
            out.append(t)
    return out, ptr


def plan():
    systems, ptr = [], 0
    for i in range(N_SYSTEMS):
        r = i // len(DOMAINS)                  # round: shifts kind and messiness so every domain gets a mix
        adv = (i + r) % 2 == 0
        types, ptr = _types(adv, ptr)
        systems.append({"system_id": f"H{i + 1:03d}", "domain": DOMAINS[i % len(DOMAINS)], "kind": KINDS[(i + r) % 3],
                        "messiness": "adversarial" if adv else "realistic", "task_types": types, "pilot": False})
    tptr = 0
    for j in range(N_TOPUP):                   # H181-H240: diagnosis + two other types, same rotations
        i = N_SYSTEMS + j
        r = i // len(DOMAINS)
        adv = (i + r) % 2 == 0
        types = ["completeness", "diagnosis"] if adv else ["diagnosis"]
        while len(types) < 3:
            t = OTHERS[tptr % len(OTHERS)]
            tptr += 1
            if t not in types:
                types.append(t)
        systems.append({"system_id": f"H{i + 1:03d}", "domain": DOMAINS[i % len(DOMAINS)], "kind": KINDS[(i + r) % 3],
                        "messiness": "adversarial" if adv else "realistic", "task_types": types, "pilot": False,
                        "topup": True})
    for j, sid in enumerate(PILOT):
        types, ptr = _types(j == 0, ptr)
        systems.append({"system_id": sid, "domain": DOMAINS[(7 + 5 * j) % len(DOMAINS)], "kind": KINDS[j % 3],
                        "messiness": "adversarial" if j == 0 else "realistic", "task_types": types, "pilot": True})
    return systems


def item_ids(system):
    return [f"{system['system_id']}-{n}" for n in (1, 2, 3)]
