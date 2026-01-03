AGENT.md — Lexi Project Agent Operating Rules

This repository contains Lexi, a complex, multi-subsystem AI product.
Automated agents (Codex, Cursor agents, SWE agents, CI bots, eval agents)
must follow these rules strictly.

These rules override agent defaults, heuristics, and inferred context.

0) Prime Directive

Do not break architectural contracts.

Lexi is composed of tightly coupled subsystems (LLM serving, memory, identity,
avatar generation, pipelines, frontend flows, orchestration).

If you are unsure whether a change preserves an existing contract:

STOP

Explain the uncertainty

Ask for confirmation

1) Authoritative Working Directory

The authoritative working directory is:

The repository root containing this AGENT.md

OR a subdirectory explicitly stated by the user at session start

You must treat this directory as:

A sealed filesystem boundary

The sole source of truth

Independent from any similarly named projects elsewhere

Do NOT infer behavior by inspecting sibling repos, backups, or prior versions.

2) Cross-Project Isolation (Hard Rule)

You MUST NOT:

Read files outside the working directory

Traverse above the repo root (..)

Inspect other Lexi copies, forks, or experiments

Pull code from:

Older Lexi versions

Gauntlet or other projects

Past challenge solutions

Any path not explicitly inside this repo

Name similarity is irrelevant. Path boundaries are absolute.

If required information is missing:

Report it

Do NOT invent or import it

3) Subsystem Boundary Rule (Critical)

Lexi subsystems must be treated as separate contracts.

Examples of subsystems:

Backend API (FastAPI routes, middleware)

Memory system (short-term, long-term, identity)

Persona / mode system

Avatar generation (Flux / ComfyUI / SD pipeline)

Frontend (React, flows, UX)

Worker / orchestration / services

Infrastructure (Docker, compose, deployment)

You MUST NOT:

Modify multiple subsystems in one change unless explicitly instructed

Refactor shared code “for cleanliness”

Move files across subsystems without approval

If a fix appears to require cross-subsystem changes:

Pause

Describe the dependency

Ask how to proceed

4) No “Helpful” Invention

Do NOT:

Invent APIs, configs, env vars, or files

Assume helpers/utilities exist unless present

Stub out missing components without being told

Create placeholder logic to “make it work”

If something is missing or unclear:

Explain exactly what is missing

Propose options

Wait for confirmation

5) Determinism, Safety, and Side Effects

No filesystem writes outside the working directory

No network access unless explicitly requested

No background services or daemons

No random behavior unless specified

No silent migrations or data rewrites

Assume:

Production data may exist

State may be persistent

Mistakes are expensive

6) Minimal Change Principle

Touch the smallest number of files possible

Prefer localized edits over refactors

Avoid renaming or moving files

Do not “clean up” unrelated code

Refactors require explicit approval.

7) Build / Test Gate

Before declaring work complete:

Identify the relevant verification steps

Run them if possible

If not possible, state exactly what would be run

Examples (only when applicable):

pnpm install --frozen-lockfile
pnpm build
pnpm test

docker compose build
docker compose up -d

pytest


Never claim verification that was not performed.

8) Git Safety Rules (Hard Constraint)
🚨 Repository Authority Rule

The local working tree is the source of truth.

The agent is allowed to:

git status

git diff

git add

git commit

git push

The agent is NOT allowed to:

git pull

git fetch

git rebase

git merge

git checkout (branch switching)

reset, rewrite, or reconcile history

These operations MUST NOT be performed unless the user explicitly and clearly
instructs the agent to do so in the current session.

If the agent believes a pull, rebase, or merge is necessary, it must:

STOP

Explain why

Ask for permission before proceeding

Violation of this rule risks data loss and is considered a critical failure.

9) Priority Order

When conflicts arise, prioritize in this order:

User instructions

This AGENT.md

Existing Lexi architectural contracts

Correctness

Performance

Cleanliness / elegance

Preserve system integrity over task completion.
