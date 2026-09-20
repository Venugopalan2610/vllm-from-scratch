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
- CI configuration (see below).

## CI

There is no CI yet. If you add one, the minimum is:

- A **CPU job** that runs the pure-logic stages (06, 09, 10, 11, 13, 14, 15,
  16, 17, 19, 20) and the harness tests.
- A **GPU job** that runs the rest, ideally on an L4 or A100.

See `dev/verify.sh` for the commands.

## Code of conduct

Be kind. Help each other. The course exists so that people learn.
