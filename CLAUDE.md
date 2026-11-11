# Deep-research workspace rules

Workspace of a research agent: receive a task, collect sources, write notes, deliver a report.
Entry point for any lookup: `INDEX.md`.

## Layout

```
CLAUDE.md                 rules (this file)
README.md                 guide for human users, Korean; exempt from writing rules; do not load for work
requirements.txt          dependency of tools/dr.py (PyYAML)
INDEX.md                  generated: task table, tag index, lookup pointers
index/catalog.tsv         generated: every object (id, type, task, status, date, kb, tags, path, title)
index/links.tsv           generated: edges src-rel-dst
index/sources.tsv         generated: cited URL -> documents
log/events.tsv            append-only work and communication log
tasks/T-0001_<slug>/      one directory per task
  task.md                 task record: request, scope, outcome
  N-0001_<slug>.md        research note (one sub-question each)
  R-0001_<slug>.md        report (deliverable)
  raw/                    optional source captures; not indexed
tools/dr.py               registry tool; the only writer of INDEX.md, index/, log/
```

Source of truth: frontmatter of `tasks/**.md` and `log/events.tsv`. Everything under `index/` and `INDEX.md` is derived; never hand-edit.

## Objects and IDs

| type | id | meaning |
|---|---|---|
| task | `T-0001` | one commissioned research job |
| note | `N-0001` | evidence collected for one sub-question |
| report | `R-0001` | synthesized deliverable |

- IDs are global, sequential, never reused. `dr.py` allocates them.
- Filenames start with the ID. Slugs: ASCII kebab-case.
- Reference other objects by ID only (`uses: [N-0003]` in frontmatter, or `N-0003` in body text). Never by path. Paths may change; IDs do not.

## Frontmatter

```yaml
---
id: N-0007
type: note                # task | note | report
task: T-0002              # owning task (note, report)
title: "..."
status: draft             # note/report: draft | final | superseded. task: active | done | dropped
created: 2026-11-11       # task: received / finished (ISO timestamps), requester, deliverable
tags: [topic-a, method-b]
uses: [N-0003]            # objects this one builds on; report lists its notes; cross-task reuse allowed
supersedes: [N-0002]      # optional; set the old object to status superseded
---
```

Tags: lowercase kebab-case, 2-4 per document, specific to that document. Check the tag list in `INDEX.md` and reuse before coining. Merge duplicates with `dr.py retag`.

## Workflow

Run as `tools/dr.py <cmd>` from the workspace root. `-h` on any command lists options.

1. **Intake.** `dr.py task "<English title>" --slug <slug> --tags a,b --request "<English brief>"`. Fill `## Scope` in `task.md`. Log each clarification: `dr.py log T-x ask "..."`, `dr.py log T-x answer "..." --actor user`.
2. **Prior-work check, before any search.** Read `INDEX.md`. `dr.py find --tag <t>` / `--text <regex>`; `grep <domain-or-title> index/sources.tsv`. Reuse existing notes through `uses`; do not re-fetch a source already extracted unless it is stale or was only verified at [A]/[M].
3. **Research.** One note per sub-question: `dr.py new note T-x <slug> --title "..."` returns the path to write. For delegated work log `dispatch` (actor `agent`, msg = sub-question) and `return` (actor `sub:<name>`, refs = note ids).
4. **Skill output.** If the deep-research skill or a subagent wrote files elsewhere (`research_notes/`, `reports/`, scratch), register them at once: `dr.py ingest note T-x <files>`, `dr.py ingest report T-x <file> --slug <slug> --uses N-a,N-b`. No unregistered `.md` stays in the workspace.
5. **Report.** `dr.py new report T-x <slug> --uses <note ids>`.
6. **Deliver and close.** `dr.py log T-x deliver "<what was sent>" --refs R-x`, then `dr.py close T-x --deliverable R-x` (drafts become final).
7. **After close.** Correction to the same deliverable: `dr.py reopen T-x "<reason>"`, log `revise`, close again. New question on the topic: new task whose documents `uses` the old ones.
8. **End of every session.** `dr.py check` must report 0 problems.

## Log

`log/events.tsv` columns: `ts task event actor refs msg`. Written only through `dr.py`; rows are never edited or deleted.

| event | when |
|---|---|
| received | task opened (automatic) |
| ask / answer | question to requester / reply |
| plan | sub-questions fixed |
| dispatch / return | work sent to / returned from a subagent |
| add | document created or ingested (automatic) |
| deliver | result handed to requester |
| revise | change after delivery |
| close / reopen | task state change (automatic) |
| admin | batch metadata edit, move (automatic; task `-`) |

Actor: `user`, `agent`, or `sub:<name>`. Msg: one line, facts only.

## Tracing and batch operations

- `dr.py show <ID>`: metadata, links out, links in, log rows of one object. Task -> documents -> log in one call.
- `dr.py find <selectors>`: `--ids --type --task --tag --status --text --url`, combined with AND.
- `dr.py set KEY=VALUE... <selectors> [--dry-run]`: edit frontmatter of every match. Example: `dr.py set status=superseded --ids N-0002,N-0004`.
- `dr.py retag OLD NEW [--dry-run]`: rename or merge a tag everywhere; NEW `-` deletes it.
- `dr.py mv <ID> --slug <new>` / `--to T-y`: rename or move; links stay valid.
- `dr.py index`: rebuild after any hand edit of frontmatter.
- Run batch edits with `--dry-run` first. Never delete documents; mark `superseded` or `dropped`.

## Writing rules

Goal: low storage and token cost. Applies to every stored file.

- English only, including titles, slugs, tags, log messages, commit messages. Exception: `README.md`. Reply to the requester in their language; store in English.
- State facts plainly. No intensifiers, no scene-setting, no restating the question, no summary that repeats the body.
- One claim per sentence or bullet, with number, unit, date, and source link.
- Mark each sourced claim: `[V]` checked against primary full text, `[A]` abstract/snippet/secondary only, `[M]` from memory, unverified. (T-0001 notes predate this and use equivalent longer labels.)
- Separate evidence from inference. Label inference as inference.
- Notes hold extracted facts and the URL, not page copies. Use `raw/` only for sources that may become unreachable.
- Tables for comparisons, bullets for lists, prose only where reasoning needs it.
- Do not duplicate content across documents; cite the ID.
- A report starts with the conclusion, then evidence. Record open questions and evidence gaps explicitly.
- `task.md` `## Outcome`: 3-5 lines on what was concluded and what remains open.
