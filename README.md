# MAE 5110 Code Assignments

Code assignments for MAE 5110. Each assignment keeps its prompt, code, models,
integrators, report, and figures in its own directory.

## Installation

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). After cloning this repository, run the following command from its root directory:

```console
uv sync --python 3.14
```

This creates a local `.venv` and installs the required dependencies. Run Python commands inside the environment with `uv run`, for example:

```console
uv run python assignments/assignment_0/assignment_0.py
```

## Assignments

- Assignment 0: [prompt](assignments/assignment_0/assignment.md) · [running the code](assignments/assignment_0/README.md)
- Assignment 1: [prompt](assignments/assignment_1/assignment.md)

## Layout

```text
assignments/
  assignment_0/    # Pendulum, bouncing ball, Euler/RK4, and existing figures
  assignment_1/    # Rimless-wheel assignment; implementation to follow
```

Run assignment scripts by file path from this repository's root. Each script
imports the `models` and `integrators` packages inside its own assignment.
The root `pyproject.toml` and local `.venv` serve all assignments.

Formal figures belong in the relevant assignment's `figures/` directory and
should be committed with the report. Use `scratch/` within an assignment for
temporary outputs; Git ignores those directories.

## Git workflow

The instructor repository is `upstream`; the personal fork is `origin` once
configured. Use `kj366/assignment_N` for assignment branches.

Before starting a new assignment, save the current work and check `git status`.
After the initial fork setup, sync your own `main` and the instructor's updates:

```bash
git switch main
git pull --ff-only origin main
git fetch upstream
git merge upstream/main
```

If a command reports conflicts or diverging histories, resolve them and check
the result before continuing. Inspect new files even if the merge succeeds:
the instructor uses different paths. Keep prompts in
`assignments/assignment_N/assignment.md`, place assignment-specific code in the
same directory, and update these links and commands as needed. Commit any path
adjustments before pushing `main`.

Then push your updated `main` and create the next assignment branch, for example:

```bash
git push origin main
git switch -c kj366/assignment_2
```

During the assignment, commit the code, report, figures, and any necessary
configuration changes. Push the assignment branch to `origin`, open a PR into
the **personal fork's `main`**, and submit the **PR link** on Canvas. Follow the
course's review schedule before merging the PR.
