# Backlog

Open questions and future work that no active task owns. Hand-edited; not indexed. Rules: `CLAUDE.md`, section Backlog.

## Open

| id | kind | opened | origin | item |
|---|---|---|---|---|
| B-0001 | work | 2026-11-11 | T-0005 review | Build a source reader that returns clean verbatim text. `tools/fulltext.py` strips HTML tags with regular expressions, so its output still carries navigation and menu text, broken table rows, math markup residue and reference lists; a PDF is only saved, because no text extractor is installed. Target: main content only, tables kept as rows, section and table captions kept, PDF text extracted in reading order, same result for arXiv, publisher pages and GitHub. Decide by measurement on about 10 already-cited sources: share of non-content characters, and whether every cited number is found with its table caption. |

## Closed

| id | kind | opened | closed | resolution |
|---|---|---|---|---|
