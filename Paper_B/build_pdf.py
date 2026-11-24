"""Build a paper PDF (and LaTeX source) from its markdown draft.

Generalised from Paper A's build script: pass --src to point it at any draft.
The markdown stays the editable artefact; this handles the title block, the
abstract, the Unicode that renders badly in a PDF, and the LaTeX plumbing.

    python build_pdf.py                       PaperB_draft.md -> .pdf and .tex
    python build_pdf.py --src ../Paper_A/PaperA_draft.md
    python build_pdf.py --no-tex

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
    ("λ", r"$\lambda$"),
    ("²", r"$^2$"),
]


def split_front_matter(text):
    title = re.search(r"^#\s+(.+)$", text, re.M)
    title = title.group(1).strip() if title else "Untitled"

    author = re.search(r"^\*\*(.+?)\*\*\s*$", text, re.M)
    author = author.group(1).strip() if author else ""

    note = re.search(r"^\*(Draft.+?)\*\s*$", text, re.M | re.S)
    note = " ".join(note.group(1).split()) if note else ""

    m = re.search(r"^##\s+Abstract\s*$(.*?)^---\s*$", text, re.M | re.S)
    abstract = m.group(1).strip() if m else ""

    start = re.search(r"^##\s+1\.", text, re.M)
    body = text[start.start():] if start else text
    return title, author, note, abstract, body


def latex_escape_meta(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "PaperB_draft.md"))
    ap.add_argument("--no-tex", action="store_true",
                    help="skip the .tex output (produced by default)")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        sys.exit("pandoc not found.")

    src = os.path.abspath(args.src)
    stem = os.path.splitext(os.path.basename(src))[0]
    outdir = os.path.dirname(src)
    pdf = os.path.join(outdir, stem + ".pdf")
    tex = os.path.join(outdir, stem + ".tex")
    tmp = os.path.join(tempfile.gettempdir(), stem + "_build.md")

    text = open(src).read()
    title, author, note, abstract, body = split_front_matter(text)
    for a, b in SUBS:
        abstract = abstract.replace(a, b)
        body = body.replace(a, b)

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
  - \\usepackage{{amsmath}}
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
    with open(tmp, "w") as f:
        f.write(meta + body)

    base = ["pandoc", tmp, "--resource-path", outdir, "--highlight-style=tango"]
    r = subprocess.run(base + ["-o", pdf, "--pdf-engine=xelatex"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        sys.exit("pandoc failed on the PDF")
    print(f"-> {pdf}  ({os.path.getsize(pdf)/1024:.0f} KB)")

    if not args.no_tex:
        r = subprocess.run(base + ["-o", tex, "--standalone"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit("pandoc failed on the LaTeX")
        print(f"-> {tex}  ({os.path.getsize(tex)/1024:.0f} KB)")

    try:
        os.remove(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    main()
