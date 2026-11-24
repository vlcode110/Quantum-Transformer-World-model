"""Build the Paper A PDF from the markdown draft.

The draft is kept as readable markdown so it can be edited without LaTeX
getting in the way. This script does the conversion: it lifts the title block
and abstract into pandoc metadata, replaces the handful of Unicode symbols that
make a mess in a PDF with proper LaTeX maths, and runs pandoc through xelatex.

    python build_pdf.py
    python build_pdf.py --keep-tex      leave the intermediate .tex for debugging

The source markdown is never modified.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "PaperA_draft.md")
OUT = os.path.join(HERE, "PaperA_draft.pdf")
# Scratch goes to the system temp directory: the project folder may be synced
# to iCloud, where stray files are both untidy and occasionally undeletable.
TMP = os.path.join(tempfile.gettempdir(), "paperA_build.md")

# Unicode that either lacks a glyph in the default fonts or simply belongs in
# maths mode. Replaced at build time only.
SUBS = [
    ("⟨Zᵢ⟩", r"$\langle Z_i \rangle$"),
    ("⟨Xᵢ⟩", r"$\langle X_i \rangle$"),
    ("⟨Z⟩", r"$\langle Z \rangle$"),
    ("⟨X⟩", r"$\langle X \rangle$"),
    ("R²", r"$R^2$"),
    ("×", r"$\times$"),
    ("≈", r"$\approx$"),
    ("−", "-"),
    ("→", r"$\rightarrow$"),
    ("²", r"$^2$"),
]


def split_front_matter(text):
    """Pull the title, author, draft note and abstract out of the body."""
    title = re.search(r"^#\s+(.+)$", text, re.M)
    title = title.group(1).strip() if title else "Untitled"

    author = re.search(r"^\*\*(.+?)\*\*\s*$", text, re.M)
    author = author.group(1).strip() if author else ""

    note = re.search(r"^\*(Draft.+?)\*\s*$", text, re.M | re.S)
    note = " ".join(note.group(1).split()) if note else ""

    m = re.search(r"^##\s+Abstract\s*$(.*?)^---\s*$", text, re.M | re.S)
    abstract = m.group(1).strip() if m else ""

    # body starts at the first numbered section
    start = re.search(r"^##\s+1\.", text, re.M)
    body = text[start.start():] if start else text
    return title, author, note, abstract, body


def latex_escape_meta(s):
    """Metadata goes through LaTeX directly, so a stray % or & would break it."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tex", action="store_true")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        sys.exit("pandoc not found — install it, or ask Claude to build the "
                 "PDF with reportlab instead.")

    text = open(SRC).read()
    title, author, note, abstract, body = split_front_matter(text)

    for a, b in SUBS:
        abstract = abstract.replace(a, b)
        body = body.replace(a, b)

    # Markdown emphasis inside the abstract would survive, but the draft note
    # sits in the author block where pandoc does not parse markdown.
    # Only the first sentence goes in the running header; the full note would
    # wrap to three lines and crowd the top margin.
    note_short = re.split(r"(?<=\.)\s", re.sub(r"[*`]", "", note))[0]
    note_clean = latex_escape_meta(note_short)

    meta = f"""---
title: "{title}"
author:
  - {author}
date: ""
abstract: |
{chr(10).join('  ' + l for l in abstract.splitlines())}
documentclass: article
classoption:
  - 11pt
geometry:
  - margin=1in
numbersections: false
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - \\usepackage{{booktabs}}
  - \\usepackage{{longtable}}
  - \\usepackage[font=small,labelfont=bf]{{caption}}
  - \\usepackage{{float}}
  - \\setlength{{\\emergencystretch}}{{3em}}
  - \\renewcommand{{\\arraystretch}}{{1.15}}
  - \\usepackage{{fancyhdr}}
  - \\pagestyle{{fancy}}
  - \\fancyhf{{}}
  - \\fancyfoot[C]{{\\thepage}}
  - \\fancyhead[L]{{\\footnotesize\\itshape {note_clean}}}
  - \\renewcommand{{\\headrulewidth}}{{0.2pt}}
---

"""

    with open(TMP, "w") as f:
        f.write(meta + body)

    cmd = ["pandoc", TMP, "-o", OUT,
           "--pdf-engine=xelatex",
           "--resource-path", HERE,
           "--highlight-style=tango",
           "-V", "fontsize=11pt"]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit("pandoc failed")

    if args.keep_tex:
        subprocess.run(["pandoc", TMP, "-o", os.path.join(HERE, "PaperA_draft.tex"),
                        "--standalone", "--resource-path", HERE])
    try:
        os.remove(TMP)
    except OSError:
        pass

    size = os.path.getsize(OUT) / 1024
    print(f"\n-> {OUT}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
