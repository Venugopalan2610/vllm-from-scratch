# Typed decision heads (formerly stage 32)

This code is **not part of the course**, and nothing checks it. It was stage
32 of the ladder until 2026-09-26. It left the ladder for two reasons:

- It does not answer a question that stage 31 measured. Every stage of the
  ladder exists because the stage before it produced a number that asks for
  it. An encoder with classification heads does not follow from a paged
  decode engine.
- It describes a commercial product ("TypeSafe Jev", with "Choice", "Score"
  and "Noul" primitives) and makes claims about it, such as "50x-200x faster
  than an LLM", that this repo cannot verify.

The files stay here unchanged, so that nothing is lost. To run their checks:

    .venv/bin/python -m pytest extras/typed_decisions/tests

The imports expect `app/s32_jev.py`. Copy `s32_jev.py` there first. The
reference solution left the `solutions` branch too. Git history keeps it:

    git show b45f1ea:.solutions/s32_jev.py
