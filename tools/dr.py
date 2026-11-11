#!/usr/bin/env python3
"""dr: registry tool for the deep-research workspace. Rules: ../CLAUDE.md

Source of truth = frontmatter of tasks/**.md + log/events.tsv.
INDEX.md and index/* are generated; never edit them by hand.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
INDEX = ROOT / "index"
LOG = ROOT / "log" / "events.tsv"
LOG_HEADER = "ts\ttask\tevent\tactor\trefs\tmsg\n"

PREFIX = {"task": "T", "note": "N", "report": "R"}
TYPE = {v: k for k, v in PREFIX.items()}
ID_RE = re.compile(r"\b[TNR]-\d{4}\b")
FM_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.S)
URL_RE = re.compile(r"https?://[^\s)\]>\"'`|]+")
STATUS = {
    "task": {"active", "done", "dropped"},
    "note": {"draft", "final", "superseded"},
    "report": {"draft", "final", "superseded"},
}
REQUIRED = {
    "task": ["id", "type", "title", "status", "requester", "received"],
    "note": ["id", "type", "task", "title", "status", "created"],
    "report": ["id", "type", "task", "title", "status", "created"],
}
EVENTS = ["received", "ask", "answer", "plan", "dispatch", "return", "add",
          "deliver", "revise", "close", "reopen", "admin"]
LIST_KEYS = {"tags", "uses", "supersedes"}
KEY_ORDER = ["id", "type", "task", "title", "status", "requester", "received",
             "finished", "created", "deliverable", "tags", "uses", "supersedes"]
PROTECTED = {"id", "type"}


# ---------- primitives ----------

def die(msg):
    sys.exit(f"dr: {msg}")


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="minutes")


def mtime_iso(path):
    return dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="minutes")


def slugify(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def ser(v):
    if isinstance(v, list):
        return "[" + ", ".join(ser(x) for x in v) + "]"
    s = str(v)
    return s if re.fullmatch(r"[A-Za-z0-9_.:+-]+", s) else json.dumps(s, ensure_ascii=False)


def csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


class Doc:
    def __init__(self, path, meta, body):
        self.path, self.meta, self.body = path, meta, body

    @property
    def id(self):
        return self.meta.get("id", "")

    @property
    def type(self):
        return self.meta.get("type", "")

    @property
    def rel(self):
        return self.path.relative_to(ROOT).as_posix()

    def lst(self, key):
        v = self.meta.get(key, [])
        return v if isinstance(v, list) else [v]

    def write(self, keep_mtime=False):
        st = self.path.stat() if keep_mtime and self.path.exists() else None
        keys = [k for k in KEY_ORDER if k in self.meta] + sorted(k for k in self.meta if k not in KEY_ORDER)
        fm = "".join(f"{k}: {ser(self.meta[k])}\n" for k in keys
                     if self.meta[k] not in ("", [], None))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"---\n{fm}---\n{self.body}", encoding="utf-8")
        if st:
            os.utime(self.path, (st.st_atime, st.st_mtime))


def parse(path):
    text = path.read_text(encoding="utf-8")
    m = FM_RE.match(text)
    if not m:
        return None
    try:
        meta = yaml.load(m.group(1), Loader=yaml.BaseLoader) or {}
    except yaml.YAMLError:
        return None
    return Doc(path, meta, text[m.end():]) if isinstance(meta, dict) else None


def scan():
    """Return (docs by id, problems). raw/ subtrees hold source captures, not documents."""
    docs, problems = {}, []
    for p in sorted(TASKS.rglob("*.md")) if TASKS.exists() else []:
        if "raw" in p.relative_to(TASKS).parts:
            continue
        d = parse(p)
        if d is None or not d.id:
            problems.append(f"unregistered (no frontmatter/id): {p.relative_to(ROOT)}")
        elif d.id in docs:
            problems.append(f"duplicate id {d.id}: {d.rel} and {docs[d.id].rel}")
        else:
            docs[d.id] = d
    return docs, problems


def read_log():
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 6:
            rows.append(dict(zip(["ts", "task", "event", "actor", "refs", "msg"], parts)))
    return rows


def log_event(task, event, msg="", actor="agent", refs="", at=None):
    LOG.parent.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text(LOG_HEADER, encoding="utf-8")
    clean = lambda s: re.sub(r"\s+", " ", str(s)).strip()
    with LOG.open("a", encoding="utf-8") as f:
        f.write("\t".join([at or now(), task, event, clean(actor), clean(refs), clean(msg)]) + "\n")


def next_id(kind, docs):
    p = PREFIX[kind]
    seen = [int(i[2:]) for i in docs if i[0] == p]
    seen += [int(i[2:]) for r in read_log() for i in ID_RE.findall(r["task"] + " " + r["refs"]) if i[0] == p]
    return f"{p}-{max(seen, default=0) + 1:04d}"


def get(docs, i, kind=None):
    d = docs.get(i)
    if d is None or (kind and d.type != kind):
        die(f"unknown {kind or 'id'}: {i}")
    return d


# ---------- graph + index ----------

def edges(docs):
    """(src, rel, dst). Declared links come from frontmatter; 'mentions' from ids in body text."""
    out = []
    for d in docs.values():
        declared = set()
        for rel in ("task", "deliverable", "uses", "supersedes"):
            for dst in d.lst(rel):
                if dst:
                    out.append((d.id, rel, dst))
                    declared.add(dst)
        for dst in sorted(set(ID_RE.findall(d.body)) - declared - {d.id}):
            out.append((d.id, "mentions", dst))
    return sorted(set(out))


def sources(docs):
    by_url = {}
    for d in docs.values():
        if d.type == "task":
            continue
        for u in set(URL_RE.findall(d.body)):
            by_url.setdefault(u.rstrip(".,;:*_"), set()).add(d.id)
    return by_url


def build(docs):
    tsv = lambda rows: "".join("\t".join(map(str, r)) + "\n" for r in rows)
    order = sorted(docs.values(), key=lambda d: (d.meta.get("task", d.id), "TRN".index(d.id[0]), d.id))
    cat = [("id", "type", "task", "status", "date", "kb", "tags", "path", "title")]
    for d in order:
        cat.append((d.id, d.type, d.meta.get("task", ""), d.meta.get("status", ""),
                    (d.meta.get("created") or d.meta.get("received", ""))[:10],
                    round(d.path.stat().st_size / 1024), ",".join(d.lst("tags")), d.rel, d.meta.get("title", "")))
    src = sources(docs)
    tags = {}
    for d in order:
        for t in d.lst("tags"):
            tags.setdefault(t, []).append(d.id)
    tasks = [d for d in order if d.type == "task"]
    n = lambda k: sum(1 for d in docs.values() if d.type == k)
    md = ["# INDEX", "",
          "Generated by `tools/dr.py index`. Do not edit.", "",
          f"{n('task')} tasks, {n('note')} notes, {n('report')} reports, {len(src)} source URLs.", "",
          "## Tasks", "",
          "| id | status | received | finished | title | deliverable | notes | dir |",
          "|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        notes = sum(1 for d in docs.values() if d.type == "note" and d.meta.get("task") == t.id)
        md.append(f"| {t.id} | {t.meta.get('status', '')} | {t.meta.get('received', '')[:10]} | "
                  f"{t.meta.get('finished', '')[:10]} | {t.meta.get('title', '')} | "
                  f"{', '.join(t.lst('deliverable'))} | {notes} | {t.path.parent.relative_to(ROOT).as_posix()}/ |")
    md += ["", "## Tags", ""]
    md += [f"- {t} ({len(ids)}): {' '.join(ids)}" for t, ids in sorted(tags.items())] or ["(none)"]
    md += ["", "## Lookup", "",
           "- `index/catalog.tsv`: every document; id, type, task, status, date, size, tags, path, title",
           "- `index/links.tsv`: edges `src rel dst` (task, deliverable, uses, supersedes, mentions)",
           "- `index/sources.tsv`: cited URL -> documents citing it",
           "- `log/events.tsv`: work and communication log",
           "- `tools/dr.py show ID`: one object with its links and log rows", ""]
    return {
        ROOT / "INDEX.md": "\n".join(md),
        INDEX / "catalog.tsv": tsv(cat),
        INDEX / "links.tsv": tsv([("src", "rel", "dst")] + edges(docs)),
        INDEX / "sources.tsv": tsv([("url", "n", "docs")] + [(u, len(i), ",".join(sorted(i))) for u, i in sorted(src.items())]),
    }


def reindex(quiet=False):
    docs, problems = scan()
    INDEX.mkdir(exist_ok=True)
    for path, content in build(docs).items():
        path.write_text(content, encoding="utf-8")
    if not quiet:
        print(f"indexed {len(docs)} objects")
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)


def select(docs, a):
    if not any([a.ids, a.type, a.task, a.tag, a.status, a.text, a.url]):
        die("selector required: --ids/--type/--task/--tag/--status/--text/--url")
    out = []
    for d in docs.values():
        if a.ids and d.id not in csv(a.ids):
            continue
        if a.type and d.type != a.type:
            continue
        if a.task and d.meta.get("task") != a.task:
            continue
        if a.tag and a.tag not in d.lst("tags"):
            continue
        if a.status and d.meta.get("status") != a.status:
            continue
        if a.text and not re.search(a.text, d.meta.get("title", "") + "\n" + d.body, re.I):
            continue
        if a.url and a.url not in d.body:
            continue
        out.append(d)
    return sorted(out, key=lambda d: d.id)


# ---------- commands ----------

def cmd_task(a):
    docs, _ = scan()
    slug = a.slug or slugify(a.title) or die("non-ASCII title: pass --slug")
    tid = next_id("task", docs)
    at = a.at or now()
    body = f"\n# {a.title}\n\n## Request\n\n{a.request or 'TODO'}\n\n## Scope\n\n## Outcome\n"
    d = Doc(TASKS / f"{tid}_{slug}" / "task.md",
            {"id": tid, "type": "task", "title": a.title, "status": "active",
             "requester": a.requester, "received": at, "tags": csv(a.tags)}, body)
    d.write()
    log_event(tid, "received", a.request or a.title, actor=a.requester, at=at)
    reindex(quiet=True)
    print(f"{tid}\t{d.rel}")


def _doc_meta(kind, did, task, title, created, a):
    return {"id": did, "type": kind, "task": task.id, "title": title, "status": a.status or "draft",
            "created": created, "tags": csv(a.tags) or task.lst("tags"), "uses": csv(a.uses)}


def cmd_new(a):
    docs, _ = scan()
    task = get(docs, a.task, "task")
    did = next_id(a.kind, docs)
    title = a.title or a.slug.replace("-", " ").capitalize()
    d = Doc(task.path.parent / f"{did}_{slugify(a.slug)}.md",
            _doc_meta(a.kind, did, task, title, now()[:10], a), f"\n# {title}\n\n")
    d.write()
    log_event(task.id, "add", title, refs=did)
    reindex(quiet=True)
    print(f"{did}\t{d.rel}")


def cmd_ingest(a):
    if a.slug and len(a.files) > 1:
        die("--slug only with a single file")
    for f in a.files:
        docs, _ = scan()
        task = get(docs, a.task, "task")
        src = Path(f).resolve()
        if not src.is_file():
            die(f"not a file: {f}")
        slug = a.slug or slugify(src.stem) or die(f"non-ASCII filename, pass --slug: {src.name}")
        old = parse(src)
        body = old.body if old else src.read_text(encoding="utf-8")
        h1 = re.search(r"^# +(.+)$", body, re.M)
        did = next_id(a.kind, docs)
        st = src.stat()
        when = mtime_iso(src)
        meta = dict(old.meta) if old else {}
        meta.update(_doc_meta(a.kind, did, task, a.title or (h1.group(1).strip() if h1 else slug), when[:10], a))
        d = Doc(task.path.parent / f"{did}_{slug}.md", meta, body if body.startswith("\n") else "\n" + body)
        d.write()
        os.utime(d.path, (st.st_atime, st.st_mtime))
        src.unlink()
        log_event(task.id, "add", f"ingested {src.name}", refs=did, at=when)
        print(f"{did}\t{d.rel}")
    reindex(quiet=True)


def cmd_log(a):
    docs, _ = scan()
    get(docs, a.task, "task")
    for r in csv(a.refs):
        get(docs, r)
    log_event(a.task, a.event, a.msg, actor=a.actor, refs=a.refs or "", at=a.at)


def cmd_close(a):
    docs, _ = scan()
    task = get(docs, a.task, "task")
    at = a.at or now()
    if a.deliverable:
        ids = csv(a.deliverable)
        for i in ids:
            get(docs, i, "report")
        task.meta["deliverable"] = ids[0] if len(ids) == 1 else ids
    if a.status == "done" and not task.meta.get("deliverable"):
        die("done task needs --deliverable R-xxxx (or use --status dropped)")
    task.meta.update(status=a.status, finished=at)
    task.write()
    n = 0
    for d in docs.values():
        if d.meta.get("task") == task.id and d.meta.get("status") == "draft" and a.status == "done":
            d.meta["status"] = "final"
            d.write(keep_mtime=True)
            n += 1
    log_event(task.id, "close", a.msg or f"{a.status}; {n} docs finalized", refs=",".join(task.lst("deliverable")), at=at)
    reindex(quiet=True)
    print(f"{task.id} {a.status}; {n} docs draft->final")


def cmd_reopen(a):
    docs, _ = scan()
    task = get(docs, a.task, "task")
    task.meta["status"] = "active"
    task.meta.pop("finished", None)
    task.write()
    log_event(task.id, "reopen", a.msg, actor=a.actor)
    reindex(quiet=True)


def cmd_index(a):
    reindex()


def cmd_check(a):
    docs, errs = scan()
    for d in docs.values():
        kind = d.type
        if kind not in PREFIX:
            errs.append(f"{d.id}: bad type '{kind}'")
            continue
        if d.id[0] != PREFIX[kind]:
            errs.append(f"{d.id}: id prefix does not match type {kind}")
        errs += [f"{d.id}: missing '{k}'" for k in REQUIRED[kind] if not d.meta.get(k)]
        if d.meta.get("status") not in STATUS[kind]:
            errs.append(f"{d.id}: bad status '{d.meta.get('status')}'")
        if kind == "task":
            if d.path.name != "task.md" or not d.path.parent.name.startswith(d.id + "_"):
                errs.append(f"{d.id}: expected tasks/{d.id}_<slug>/task.md, found {d.rel}")
            if d.meta.get("status") == "done" and not d.meta.get("deliverable"):
                errs.append(f"{d.id}: done without deliverable")
            if d.meta.get("status") != "active" and not d.meta.get("finished"):
                errs.append(f"{d.id}: closed without 'finished'")
        else:
            if not d.path.name.startswith(d.id + "_"):
                errs.append(f"{d.id}: filename must start with '{d.id}_': {d.rel}")
            t = docs.get(d.meta.get("task", ""))
            if t and d.path.parent != t.path.parent:
                errs.append(f"{d.id}: not inside directory of {t.id}")
    for s, rel, dst in edges(docs):
        if dst not in docs:
            errs.append(f"{s}: dangling {rel} -> {dst}")
        elif rel == "task" and docs[dst].type != "task":
            errs.append(f"{s}: task -> {dst} is not a task")
        elif rel == "deliverable" and (docs[dst].type != "report" or docs[dst].meta.get("task") != s):
            errs.append(f"{s}: deliverable {dst} is not a report of this task")
    logged = set()
    for r in read_log():
        logged.add(r["task"])
        if r["event"] not in EVENTS:
            errs.append(f"log {r['ts']}: unknown event '{r['event']}'")
        errs += [f"log {r['ts']}: unknown id {i}" for i in ID_RE.findall(r["task"] + " " + r["refs"]) if i not in docs]
    errs += [f"{i}: no log rows" for i, d in docs.items() if d.type == "task" and i not in logged]
    for path, content in build(docs).items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            errs.append(f"stale index: {path.relative_to(ROOT)} (run: tools/dr.py index)")
    for e in errs:
        print(e)
    print(f"{len(docs)} objects, {len(errs)} problems")
    sys.exit(1 if errs else 0)


def cmd_show(a):
    docs, _ = scan()
    d = get(docs, a.id)
    print(d.rel)
    for k in [k for k in KEY_ORDER if k in d.meta]:
        print(f"  {k}: {ser(d.meta[k])}")
    es = edges(docs)
    title = lambda i: docs[i].meta.get("title", "") if i in docs else "(missing)"
    for label, rows in (("out", [(r, t) for s, r, t in es if s == d.id]),
                        ("in", [(r, s) for s, r, t in es if t == d.id])):
        for rel, other in rows:
            print(f"  {label} {rel:<11} {other}  {title(other)}")
    for r in read_log():
        if r["task"] == d.id or d.id in csv(r["refs"]):
            print(f"  log {r['ts']} {r['event']:<8} {r['actor']}  {r['refs']}  {r['msg']}")


def cmd_find(a):
    docs, _ = scan()
    for d in select(docs, a):
        print(f"{d.id}\t{d.rel}\t{d.meta.get('title', '')}")


def cmd_set(a):
    docs, _ = scan()
    pairs = []
    for kv in a.assign:
        k, sep, v = kv.partition("=")
        if not sep or k in PROTECTED:
            die(f"bad assignment: {kv}")
        pairs.append((k, csv(v) if k in LIST_KEYS else v))
    hits = select(docs, a)
    for d in hits:
        for k, v in pairs:
            if k == "status" and v not in STATUS.get(d.type, ()):
                die(f"{d.id}: status '{v}' invalid for {d.type}")
            print(f"{d.id}\t{k}: {ser(d.meta.get(k, ''))} -> {ser(v)}")
            d.meta[k] = v
        if not a.dry_run:
            d.write(keep_mtime=True)
    if not a.dry_run and hits:
        log_event("-", "admin", "set " + " ".join(a.assign), refs=",".join(d.id for d in hits))
        reindex(quiet=True)
    print(f"{len(hits)} objects{' (dry run)' if a.dry_run else ''}")


def cmd_retag(a):
    docs, _ = scan()
    hits = [d for d in docs.values() if a.old in d.lst("tags")]
    for d in sorted(hits, key=lambda d: d.id):
        tags = [a.new if t == a.old else t for t in d.lst("tags")]
        d.meta["tags"] = [t for i, t in enumerate(tags) if t != "-" and t not in tags[:i]]
        print(f"{d.id}\ttags: {ser(d.meta['tags'])}")
        if not a.dry_run:
            d.write(keep_mtime=True)
    if not a.dry_run and hits:
        log_event("-", "admin", f"retag {a.old} -> {a.new}", refs=",".join(sorted(d.id for d in hits)))
        reindex(quiet=True)
    print(f"{len(hits)} objects{' (dry run)' if a.dry_run else ''}")


def cmd_mv(a):
    docs, _ = scan()
    d = get(docs, a.id)
    if d.type == "task":
        if not a.slug or a.to:
            die("tasks: only --slug")
        new = TASKS / f"{d.id}_{slugify(a.slug)}"
        d.path.parent.rename(new)
        print(new.relative_to(ROOT))
    else:
        task = get(docs, a.to or d.meta["task"], "task")
        slug = slugify(a.slug) if a.slug else d.path.stem.split("_", 1)[1]
        new = task.path.parent / f"{d.id}_{slug}.md"
        d.path.rename(new)
        d.path = new
        if a.to:
            d.meta["task"] = task.id
            d.write(keep_mtime=True)
        print(d.rel)
    log_event("-", "admin", f"mv {a.id}" + (f" to {a.to}" if a.to else "") + (f" slug {a.slug}" if a.slug else ""), refs=a.id)
    reindex(quiet=True)


def main():
    ap = argparse.ArgumentParser(prog="dr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def selector(p):
        for flag in ("--ids", "--type", "--task", "--tag", "--status", "--text", "--url"):
            p.add_argument(flag)

    def docopts(p):
        p.add_argument("--title")
        p.add_argument("--tags", help="comma list; default: task tags")
        p.add_argument("--uses", help="comma list of ids this document builds on")
        p.add_argument("--status", choices=sorted(STATUS["note"]))

    p = sub.add_parser("task", help="open a task (commission)")
    p.add_argument("title")
    p.add_argument("--slug")
    p.add_argument("--requester", default="user")
    p.add_argument("--request", help="English brief of the request")
    p.add_argument("--tags")
    p.add_argument("--at", help="ISO timestamp override (backfill)")
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser("new", help="create an empty note/report in a task")
    p.add_argument("kind", choices=["note", "report"])
    p.add_argument("task")
    p.add_argument("slug")
    docopts(p)
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("ingest", help="move existing files into a task and register them")
    p.add_argument("kind", choices=["note", "report"])
    p.add_argument("task")
    p.add_argument("files", nargs="+")
    p.add_argument("--slug")
    docopts(p)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("log", help="append a work/communication event")
    p.add_argument("task")
    p.add_argument("event", choices=EVENTS)
    p.add_argument("msg")
    p.add_argument("--actor", default="agent", help="user | agent | sub:<name>")
    p.add_argument("--refs", help="comma list of ids")
    p.add_argument("--at")
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("close", help="finish a task; drafts become final")
    p.add_argument("task")
    p.add_argument("--deliverable", help="report id, or comma list when several reports were delivered")
    p.add_argument("--status", choices=["done", "dropped"], default="done")
    p.add_argument("--msg")
    p.add_argument("--at")
    p.set_defaults(fn=cmd_close)

    p = sub.add_parser("reopen", help="set a closed task active again")
    p.add_argument("task")
    p.add_argument("msg")
    p.add_argument("--actor", default="user")
    p.set_defaults(fn=cmd_reopen)

    sub.add_parser("index", help="rebuild INDEX.md and index/*").set_defaults(fn=cmd_index)
    sub.add_parser("check", help="validate metadata, links, log, index freshness").set_defaults(fn=cmd_check)

    p = sub.add_parser("show", help="one object: metadata, links in/out, log rows")
    p.add_argument("id")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("find", help="list objects matching all given selectors")
    selector(p)
    p.set_defaults(fn=cmd_find)

    p = sub.add_parser("set", help="batch-edit frontmatter: set KEY=VALUE... <selectors>")
    p.add_argument("assign", nargs="+", metavar="KEY=VALUE")
    selector(p)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_set)

    p = sub.add_parser("retag", help="rename/merge a tag everywhere; NEW '-' deletes")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_retag)

    p = sub.add_parser("mv", help="rename slug or move a document to another task; ids keep links valid")
    p.add_argument("id")
    p.add_argument("--slug")
    p.add_argument("--to", metavar="TASK")
    p.set_defaults(fn=cmd_mv)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
