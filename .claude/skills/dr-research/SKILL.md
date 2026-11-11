---
name: dr-research
description: Research workflow of this workspace; replaces the generic deep-research skill here. Use for any request to research, compare, survey, verify or update a topic, and for questions that earlier tasks may already answer. Runs intake, prior-work check, tiered research (lookup, quick, standard, deep), report and close through tools/dr.py.
argument-hint: "[lookup|quick|standard|deep] [researcher=<model>] [writer=<model>] [verifier=<model>] <request>"
---

# dr-research

Request: $ARGUMENTS

`CLAUDE.md` holds the registry and writing rules; this file adds only the research procedure. `ROOT` = workspace root (absolute). Run `tools/dr.py` from `ROOT`.

## Role

You coordinate: registry commands, decomposition, dispatch, delivery. You are also the only reader who sees the request, every note and the report together, so three judgments are yours and are not delegated:

1. Scope and decomposition (steps 3-4): what the request means, how many notes it needs, which tier.
2. Evidence review (step 7): read every note in full, check the claims the conclusion will rest on against their sources, fix the position the report takes.
3. Acceptance (step 10): the report says what the notes support, no more.

In `standard` and `deep`, do not run the literature search and do not write note or report bodies; subagents do that. Fetching a cited source to check a claim is part of review, not search. In `quick`, or when no subagent tool exists, read `researcher.md` and `report-writer.md` and do the work yourself.

## Tiers

Use the tier the requester names. Otherwise pick the lowest tier that answers the request. If a tier proves too low, escalate once and log why.

| tier | use when | researchers | search+fetch calls each | gap round | verify pass | note / report target |
|---|---|---|---|---|---|---|
| lookup | registry documents already answer it | 0 | 0 | no | no | no new document |
| quick | one narrow question; low cost of error | 0 (you) | 8 | no | no | 4 kB / 3 kB |
| standard | default; 2-4 notes | 2-4 | 15 | only if a gap blocks the conclusion | no | 6 kB / 8 kB |
| deep | requester asks for depth, costly decision, or 5+ notes | 5-8 | 25 | one | yes | 10 kB / 15 kB |

Size targets bound prose, not evidence: cut wording before cutting a sourced fact.

Pick the tier after decomposing (step 4), by the number of notes the request needs, not by how short the request reads. A survey of several method families is `deep` even when asked in one sentence. If the note count exceeds the tier the requester named, say so and offer: narrow the scope, or raise the tier.

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

- Read `INDEX.md` and `BACKLOG.md` (a backlog row the request resolves is cited in the task and closed at delivery). Run `tools/dr.py find --tag <t>` and `--text <regex>` for the request's key terms; `grep -i <domain-or-title> index/sources.tsv` for sources the requester named.
- Classify the request:
  - Correction of a delivered report: `tools/dr.py reopen T-x "<reason>"`, fix, log `revise`, close again. No new task.
  - Answered by existing documents, evidence current and `[V]` for the key claims: tier `lookup`.
  - New or partly covered: new task; list reusable note IDs for `uses`.

### 2. Lookup tier

Read only the matching documents. Reply in the requester's language with the answer, the IDs, and the evidence date. Then `tools/dr.py log T-x answer "lookup: <question> -> <one-line answer>" --refs <ids>` on the task that owns the cited report. Stop here. If the key claim is `[A]`/`[M]` or older than the request tolerates, escalate to `quick` instead.

### 3. Intake

- `tools/dr.py task "<English title>" --slug <slug> --tags a,b --request "<English brief>" [--requester <name>]`.
- Ask only when the answer changes what is researched: ambiguous entity, region, time range, or a term with two or more technical readings that lead to different sub-questions. One round, each question with a stated default; if no answer is possible, proceed on the default. Requester is a person: AskUserQuestion. Requester is an agent: message it, or proceed on a stated assumption. Log `ask` / `answer` (`--actor user`). Write assumptions into `## Scope`.
- Fill `## Scope` in `task.md`: tier, sub-questions, constraints, requested deliverable format, reused IDs.

### 4. Plan

Split the request into sub-questions that do not overlap and together cover it; drop what reused notes already answer at `[V]`. One researcher per sub-question. Rules:

- At most 3 key questions per note. A fourth means a second note. A researcher with 6 key questions and 15 calls answers each from one abstract.
- The literal subject of the request gets its own note; never one bullet inside a neighbouring note.
- Name the baseline each method is compared against and the metric that counts for the requester (for example DER, not SI-SDR).
- Say which key questions may have no published answer, so the researcher records "not studied" early instead of spending the budget on it.

For each:

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

Dispatch again only for a gap without which the conclusion cannot be stated. If the return messages do not settle that, decide it during the evidence review (step 7). New sub-question = new note. Other gaps go to the report's open questions.

### 7. Evidence review (standard and deep; you, not a subagent)

Researchers and the writer each see one slice; errors that survive are the ones only a cross-reader catches. Before any report is written:

1. Read every note in full. Do not work from the return messages.
2. List the load-bearing claims: the numbers the conclusion and any recommendation will rest on (at most 8).
3. Check each against its source with `tools/fulltext.py <url> -k <number>`, whatever its marker: metric name, unit, dataset and split, baseline, direction. A number under the wrong metric is the commonest error. Raise the marker to `[V]` when the verbatim text matches; correct the note when it does not, with a dated correction line.
4. Look for what a careful reader would doubt: figures too regular, abstract and table disagreeing, a baseline too weak or too old to transfer, one source carrying a whole conclusion.
5. Separate "tested, no gain" from "not studied". Only the first supports "does not help".
6. Decide the gap round (step 6 rule) now if not yet used.
7. Write the position brief for the writer, 5-10 lines: the answer, the claims it rests on (note ID + number), what must not be overstated, requester terms that need a one-line definition.

Log: `tools/dr.py log T-x plan "review: <n> claims checked, <n> corrected; position: <one line>"`.

### 8. Report

`tools/dr.py new report T-x <slug> --title "<working title>" --uses <all note ids, including reused>`. Dispatch one writer (Agent `model` = writer model):

```
Write report R-x answering: <full question, as amended by any follow-up>
Notes: <absolute paths of this task's notes>
Reused earlier work: <IDs and paths, or "none">
Position brief (from the coordinator's evidence review; follow it, or state where the notes contradict it):
<5-10 lines>
Deliverable format requested: <format from the brief, or "default">
Target: <n> kB.
Report file (exists, has frontmatter; fill the body, set the title): <absolute path of R-x>
First read ROOT/.claude/skills/dr-research/report-writer.md and follow it.
```

### 9. Verify pass (deep only)

Covers the `[A]`/`[M]` claims the evidence review did not reach.

Dispatch one verifier (Agent `model` = verifier model; log `dispatch` / `return`, actor `sub:verify`):

```
Verify R-x at <path>. Take the claims the conclusion depends on that are marked [A] or [M] (max 12).
For each: read the primary source with ROOT/tools/fulltext.py (verbatim text, not a summarizing fetch), compare the number, metric, unit, date and wording.
Confirmed -> change the marker to [V] in the report and in the source note. Wrong -> correct it in both and say so.
Unreachable -> leave the marker, add the claim to "Open questions and gaps".
Do not touch frontmatter. Return a table: claim, old marker, result, URL.
```

### 10. Deliver and close

1. Acceptance. Read the report against the notes and the position brief:
   - every conclusion bullet and every ranked recommendation traces to a claim in a note, with the same metric, direction and marker;
   - "no gain" is written only where a source tested it; otherwise "untested";
   - the title claims no more than the weakest conclusion bullet;
   - it answers the question as the requester asked it, in the requester's terms.
   Small faults: fix them yourself. A wrong position: send it back to the writer with the reason.
2. Fill `## Outcome` in `task.md` (3-5 lines: conclusions, what remains open).
3. Hand over the report: file-sending tool if one exists, else its path. Reply in the requester's language: 3-5 sentences, the `R-id`, the main open question, and which conclusions rest on `[A]` or single-source evidence. Do not store a translation.
4. `tools/dr.py log T-x deliver "<what was sent>" --refs R-x`
5. `tools/dr.py close T-x --deliverable R-x`
6. An open question that would change the conclusion if answered: add a row to `BACKLOG.md`. Close any backlog row this task resolved.
7. `tools/dr.py check` -> 0 problems. Files written outside `tasks/` by any subagent: `tools/dr.py ingest`.
