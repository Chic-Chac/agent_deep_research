#!/usr/bin/env python3
"""Fetch a source as verbatim text and show keyword contexts.

Use instead of a summarizing fetch tool when a claim needs a [V] marker:
the output is the page's own text, not a model's summary.

  tools/fulltext.py URL                      head of the text + saved path
  tools/fulltext.py URL -k 17.58 -k "Table 3"  contexts around each keyword
  tools/fulltext.py URL --all                whole text to stdout

arXiv /abs/ and /pdf/ URLs are read from the HTML rendering when one exists.
A PDF is saved to disk; open the printed path with the Read tool.
Nothing is written inside the workspace (cache: $TMPDIR/dr-fulltext).
"""
import argparse
import hashlib
import html
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

UA = "Mozilla/5.0 (compatible; research-fetch)"
ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(v\d+)?", re.I)


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def candidates(url):
    m = ARXIV.search(url)
    if not m:
        return [url]
    aid, ver = m.group(1), m.group(2) or ""
    return [
        f"https://arxiv.org/html/{aid}{ver}",
        f"https://ar5iv.labs.arxiv.org/html/{aid}",
        f"https://arxiv.org/pdf/{aid}{ver}",
    ]


def to_text(raw):
    t = raw.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style|annotation|annotation-xml)\b.*?</\1>", " ", t)
    t = re.sub(r"(?i)</t[dh]>", " | ", t)
    t = re.sub(r"(?i)</(tr|p|div|li|h[1-6]|caption|figcaption)>|<br\s*/?>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t ​]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    return re.sub(r"\n?\|\n", " | ", t).strip()  # keep table rows on one line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("-k", "--keyword", action="append", default=[], help="literal string to locate; repeatable")
    ap.add_argument("-w", "--width", type=int, default=300, help="characters of context on each side (default 300)")
    ap.add_argument("-n", "--max-hits", type=int, default=3, help="contexts shown per keyword (default 3)")
    ap.add_argument("--all", action="store_true", help="print the whole text")
    a = ap.parse_args()

    cache = os.path.join(tempfile.gettempdir(), "dr-fulltext")
    os.makedirs(cache, exist_ok=True)
    raw = ctype = used = None
    errors = []
    for u in candidates(a.url):
        try:
            raw, ctype = get(u)
        except Exception as e:  # network errors, HTTP errors
            errors.append(f"{u}: {e}")
            continue
        is_pdf = raw[:5] == b"%PDF-"
        # ar5iv answers a missing paper with a short stub page, not a 404
        if not is_pdf and len(raw) < 5000 and u != a.url:
            errors.append(f"{u}: stub page ({len(raw)} bytes)")
            raw = None
            continue
        used = u
        break
    if raw is None:
        sys.exit("fetch failed\n" + "\n".join(errors))

    stem = os.path.join(cache, hashlib.sha1(used.encode()).hexdigest()[:12])
    if raw[:5] == b"%PDF-":
        path = stem + ".pdf"
        with open(path, "wb") as f:
            f.write(raw)
        try:
            text = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
        except (OSError, subprocess.CalledProcessError):
            print(f"source: {used}\nPDF saved: {path}\nNo pdftotext here: open that path with the Read tool (pages=\"1-10\" for long files).")
            return
    else:
        text = to_text(raw)
    path = stem + ".txt"
    with open(path, "w") as f:
        f.write(text)
    print(f"source: {used}\ntext: {path} ({len(text)} chars)")

    if a.all:
        print(text)
        return
    if not a.keyword:
        print(text[:1500])
        print("...\n(pass -k KEYWORD for contexts, --all for everything, or grep/Read the text file)")
        return
    for k in a.keyword:
        hits = [m.start() for m in re.finditer(re.escape(k), text)]
        print(f"\n## {k!r}: {len(hits)} hit(s)")
        for i in hits[: a.max_hits]:
            print("--- " + text[max(0, i - a.width): i + len(k) + a.width].replace("\n", " / "))


if __name__ == "__main__":
    main()
