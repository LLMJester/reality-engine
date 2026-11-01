"""Run 5 hand-written-style items (V3-042, docs/RUN5_DESIGN.md section 5): 500 items written and checked by Claude
Sonnet following the v2 construction protocol (rpp-core/data/BUILDER_PROMPT.md), keys computed by a hidden
simulator run in the Docker sandbox.

  plan.py      the balanced assignment of systems (domain, kind, messiness, three task types); code only
  writer.py    the writer's tool server (save a file, run the simulator in Docker, submit) and the writer brief
  factory.py   runs writers and checks (validator, judge, leak check, adjudicator); resumable; writes items/hand5

Separation: writers and checkers run headless with no file, shell or web access; they never see a design. Design
writers never see these items. The v2 held-out split is never opened.
"""
