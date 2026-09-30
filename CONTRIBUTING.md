# Contributing

Thank you for your interest in improving this course.

## How to contribute

1. **Fork** the repo and create a feature branch.
2. **Run the checks** before you push: `dev/verify.sh` runs the full suite
   against the reference solutions.
3. **Open a pull request** with a clear description of what you changed and why.

## What helps most

- Bug reports with a failing check name and the output.
- New test cases, especially for edge cases in the scheduler or prefix cache.
- Typo fixes and documentation improvements.
- A GPU job for CI (see below).

## Rules for new course material

`docs/METHOD.md` ends with the rules for authors. In short:

- Put a prediction cell before each measurement, and name the move.
- Give the plain words first, and the GPU term second.
- Give the simple tool before the formal one: the doubling test before a fit.
- Name each variable by what it is, of what, and its unit. A ratio is
  `<top>_over_<bottom>`. `dev/names.py` checks this.
- Each stage in `stages.yaml` has a `layer:` and a `because:`.
  `tests/test_docs.py` checks them, and checks that the docs agree with the
  ladder.
- Run `dev/jargon.py` and `dev/names.py` after you change a notebook. The
  harness tests run both.

## CI

`.github/workflows/ci.yml` runs one CPU job on each push and pull request.
The job runs:

- The pure-logic stages (06, 09, 10, 11, 13, 14, 15, 16, 17, 19, 20) on a
  CPU, against the reference solutions.
- The harness tests.

There is no GPU job yet. A GPU job for the other stages, ideally on an L4 or
an A100, is a welcome contribution. See `dev/verify.sh` for the commands.

## Code of conduct

Be kind. Help each other. The course exists so that people learn.
