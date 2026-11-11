---
name: dr-research
description: Research workflow of this workspace; replaces the generic deep-research skill here. Use for any request to research, compare, survey, verify or update a topic, and for questions that earlier tasks may already answer. Runs intake, prior-work check, tiered research (lookup, quick, standard, deep), report and close through tools/dr.py.
argument-hint: "[lookup|quick|standard|deep] [researcher=<model>] [writer=<model>] [verifier=<model>] <request>"
---

# dr-research

Request: $ARGUMENTS

`CLAUDE.md` holds the registry and writing rules; this file adds only the research procedure. `ROOT` = workspace root (absolute). Run `tools/dr.py` from `ROOT`.

## Role

You coordinate: registry commands, decomposition, dispatch, delivery. In `standard` and `deep`, do not search, do not write note or report bodies, and do not read `researcher.md` or `report-writer.md`; subagents do that. In `quick`, or when no subagent tool exists, read both files and do the work yourself.

## Tiers

Use the tier the requester names. Otherwise pick the lowest tier that answers the request. If a tier proves too low, escalate once and log why.

| tier | use when | researchers | search+fetch calls each | gap round | verify pass | note / report target |
|---|---|---|---|---|---|---|
| lookup | registry documents already answer it | 0 | 0 | no | no | no new document |
| quick | one narrow question; low cost of error | 0 (you) | 8 | no | no | 4 kB / 3 kB |
| standard | default; 2-4 sub-questions | 2-4 | 15 | only if a gap blocks the conclusion | no | 6 kB / 8 kB |
| deep | requester asks for depth, costly decision, or 5+ facets/entities | 5-8 | 25 | one | yes | 10 kB / 15 kB |

Size targets bound prose, not evidence: cut wording before cutting a sourced fact.

## Models

Pass the role's model as the `model` argument of every Agent call.

| role | default | override key |
|---|---|---|
| researcher | sonnet | `researcher=` |
| report writer | opus | `writer=` |
| verifier | opus | `verifier=` |

- Values: `haiku`, `sonnet`, `opus`, `fable`.
- Override sources, highest first: `key=value` in the request arguments; the same stated in the request text or task brief by the requester (person or agent); the defaults above.
- You may change a role's model yourself only when the default cannot do the sub-question (for example a researcher that must reason over long primary texts); state the reason in the `plan` log.
- An override applies to this task only. Record non-default models in the `plan` log line.
- `quick` and `lookup` spawn no subagents; the keys have no effect there.

## Procedure

### 1. Prior-work check (every tier, before intake)

- Read `INDEX.md`. Run `tools/dr.py find --tag <t>` and `--text <regex>` for the request's key terms; `grep -i <domain-or-title> index/sources.tsv` for sources the requester named.
- Classify the request:
  - Correction of a delivered report: `tools/dr.py reopen T-x "<reason>"`, fix, log `revise`, close again. No new task.
  - Answered by existing documents, evidence current and `[V]` for the key claims: tier `lookup`.
  - New or partly covered: new task; list reusable note IDs for `uses`.

### 2. Lookup tier

Read only the matching documents. Reply in the requester's language with the answer, the IDs, and the evidence date. Then `tools/dr.py log T-x answer "lookup: <question> -> <one-line answer>" --refs <ids>` on the task that owns the cited report. Stop here. If the key claim is `[A]`/`[M]` or older than the request tolerates, escalate to `quick` instead.

### 3. Intake

- `tools/dr.py task "<English title>" --slug <slug> --tags a,b --request "<English brief>" [--requester <name>]`.
- Ask only when the answer changes what is researched (ambiguous entity, region, time range). Requester is a person: AskUserQuestion. Requester is an agent: message it, or proceed on a stated assumption. Log `ask` / `answer` (`--actor user`). Write assumptions into `## Scope`.
- Fill `## Scope` in `task.md`: tier, sub-questions, constraints, requested deliverable format, reused IDs.

### 4. Plan

Split the request into sub-questions that do not overlap and together cover it; drop what reused notes already answer at `[V]`. One researcher per sub-question, count per the tier table. For each:

`tools/dr.py new note T-x <slug> --title "<sub-question>" --tags a,b` -> prints `N-id` and path.

Then `tools/dr.py log T-x plan "<tier>; N-a: ..., N-b: ...; reuses N-c[; models researcher=..., writer=...]"`.

### 5. Dispatch researchers (all in one turn, foreground)

Log each: `tools/dr.py log T-x dispatch "<sub-question>" --refs N-a`. Agent `model` = researcher model. Prompt:

```
Research: <narrow sub-question>
Objective: <what the note must establish>
Key questions:
- <q1>
- <q2>
Suggested sources: <source types to prefer>
Constraints: <time range, region, stack limits, anything from the request>
Prior work: <N-ids to read first, or "none">. URLs in ROOT/index/sources.tsv are already extracted.
Budget: <n> search+fetch calls. Note target: <n> kB.
Note file (exists, has frontmatter; fill the body): <absolute path of N-a>
First read ROOT/.claude/skills/dr-research/researcher.md and follow it.
```

On return: `tools/dr.py log T-x return "<one-line result>" --actor sub:<slug> --refs N-a`, then `tools/dr.py index && tools/dr.py check`. A note reported as `unregistered` lost its frontmatter: restore it from `git` or re-create with `dr.py new note` and move the body.

### 6. Gap round (per tier table, at most once)

Dispatch again only for a gap without which the conclusion cannot be stated. New sub-question = new note. Other gaps go to the report's open questions.

### 7. Report

`tools/dr.py new report T-x <slug> --title "<working title>" --uses <all note ids, including reused>`. Dispatch one writer (Agent `model` = writer model):

```
Write report R-x answering: <full question, as amended by any follow-up>
Notes: <absolute paths of this task's notes>
Reused earlier work: <IDs and paths, or "none">
Deliverable format requested: <format from the brief, or "default">
Target: <n> kB.
Report file (exists, has frontmatter; fill the body, set the title): <absolute path of R-x>
First read ROOT/.claude/skills/dr-research/report-writer.md and follow it.
```

### 8. Verify pass (deep only)

Dispatch one verifier (Agent `model` = verifier model; log `dispatch` / `return`, actor `sub:verify`):

```
Verify R-x at <path>. Take the claims the conclusion depends on that are marked [A] or [M] (max 12).
For each: fetch the primary source, compare the number, unit, date and wording.
Confirmed -> change the marker to [V] in the report and in the source note. Wrong -> correct it in both and say so.
Unreachable -> leave the marker, add the claim to "Open questions and gaps".
Do not touch frontmatter. Return a table: claim, old marker, result, URL.
```

### 9. Deliver and close

1. Read the report. Fill `## Outcome` in `task.md` (3-5 lines: conclusions, what remains open).
2. Hand over the report: file-sending tool if one exists, else its path. Reply in the requester's language: 3-5 sentences, the `R-id`, and the main open question. Do not store a translation.
3. `tools/dr.py log T-x deliver "<what was sent>" --refs R-x`
4. `tools/dr.py close T-x --deliverable R-x`
5. `tools/dr.py check` -> 0 problems. Files written outside `tasks/` by any subagent: `tools/dr.py ingest`.
