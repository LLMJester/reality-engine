# Grading rubric (frozen; hash in PREREG.LOCK)

You grade one answer to one question about a system. You receive the system description, the question, the answer key (a summary, weighted key points, and known wrong answers), and the answer to grade. Grade only against the key; do not reward style, length or confidence.

For **each key point**, decide:
- `hit` — the answer clearly states this point (in any wording) and gets any quantity right within the tolerance stated in the key point (if no tolerance is stated: exact for counts and discrete states, ±5% for continuous quantities).
- `partial` — the answer states the point but with a material inaccuracy (wrong magnitude beyond tolerance but right direction, right cause but wrong mechanism, right conflict but wrong location), or states it only vaguely.
- `miss` — absent, or contradicted by the answer.

Then list **material errors**: claims in the answer that contradict the key or the description and bear on the question — e.g. naming a wrong primary cause, asserting a conflict that does not exist, predicting a wrong value for something the question asks about, or presenting a known wrong answer as the answer. Do not count harmless extra detail. Listing several alternative answers to hedge (e.g. several candidate causes, several possible values) counts each alternative that contradicts the key as a material error, unless it is explicitly ruled out in the answer.

Task-specific notes:
- **diagnosis**: if the answer commits to a wrong primary cause, the cause key point is a `miss` even if the right cause is mentioned as a secondary possibility; if it lists several causes without committing, the cause key point is at most `partial`.
- **consistency**: each injected conflict is a key point; each conflict the answer asserts that is not real is a material error.
- **prediction / intervention**: each value or consequence is a key point; numbers outside tolerance are `partial` at best if the direction is right, `miss` if the direction is wrong.
- **assumptions / completeness**: an assumption or omission counts as `hit` if it is the same condition in substance, even if phrased differently or more generally, as long as it would be the deciding factor in the same way.

Return JSON only: `{"key_points": [{"id": "...", "verdict": "hit|partial|miss", "reason": "..."}], "material_errors": ["..."], "comment": "..."}`.

**Score** (computed by code, not by you): weighted mean over key points with hit = 1, partial = 0.5, miss = 0, minus 0.1 per material error (no cap), floored at 0. **Rubric level** = floor(score × 4 + 0.5) (0–4).
