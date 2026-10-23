"""report.md -> report.html -> PDF (Microsoft Edge headless 인쇄)."""
import subprocess
import sys
import time
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: 'Malgun Gothic', sans-serif; font-size: 10.2pt; line-height: 1.6; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4px; }
.sub { text-align: center; color: #444; margin-bottom: 14px; }
h2 { font-size: 12.5pt; border-bottom: 1.5px solid #333; padding-bottom: 2px; margin-top: 18px; }
h3 { font-size: 11pt; margin: 12px 0 4px; }
table { border-collapse: collapse; margin: 8px auto; font-size: 9pt; }
th, td { border: 1px solid #999; padding: 3px 7px; text-align: center; }
th { background: #eee; }
img { display: block; margin: 8px auto; max-width: 92%; }
.cap { text-align: center; font-size: 9pt; color: #333; margin-top: -2px; }
blockquote { border-left: 3px solid #888; margin: 8px 0; padding: 4px 10px; background: #f6f6f6; }
code { font-size: 9pt; }
.abstract { border: 1px solid #aaa; padding: 8px 12px; background: #fafafa; }
"""


def main(name="report"):
    md = (HERE / f"{name}.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "md_in_html"])
    html = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    out_html = HERE / f"{name}.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = HERE / f"{name}.pdf"
    before = pdf.stat().st_mtime if pdf.exists() else 0
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, timeout=120)
    for _ in range(60):  # Edge는 프로세스 종료 후 비동기로 파일을 쓴다
        if pdf.exists() and pdf.stat().st_mtime > before:
            break
        time.sleep(1)
    print("wrote", pdf)


if __name__ == "__main__":
    main(*sys.argv[1:])
