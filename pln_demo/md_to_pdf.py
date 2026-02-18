#!/usr/bin/env python3
"""Convert DEMO_GUIDE.md to a styled PDF using fpdf2's markdown support."""

import os
import re
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(SCRIPT_DIR, "DEMO_GUIDE.md")
PDF_PATH = os.path.join(SCRIPT_DIR, "DEMO_GUIDE.pdf")


class PLNGuidePDF(FPDF):
    """Custom PDF with PLN branding, headers, footers, and markdown rendering."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "PLN x TerraMind -- Demo Guide", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Confidential -- PLN x IBM TerraMind", align="C")

    def cover_page(self):
        self.add_page()
        self.ln(50)

        # Title
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(26, 82, 118)  # #1a5276
        self.cell(0, 15, "PLN x TerraMind", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(5)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(86, 101, 115)
        self.cell(0, 10, "Demo Guide", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(3)
        self.set_font("Helvetica", "I", 12)
        self.cell(0, 8, "Satellite-Powered Infrastructure Intelligence", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "for Perusahaan Listrik Negara, Indonesia", align="C", new_x="LMARGIN", new_y="NEXT")

        self.ln(15)
        self.set_draw_color(46, 134, 193)  # #2e86c1
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())

        self.ln(15)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(100, 100, 100)

        lines = [
            "Powered by IBM TerraMind (ICCV 2025)",
            "Any-to-Any Earth Observation Foundation Model",
            "",
            "For presenters and non-technical stakeholders",
            "Includes: Dashboard walkthrough, scenario objectives, full glossary",
        ]
        for line in lines:
            self.cell(0, 7, line, align="C", new_x="LMARGIN", new_y="NEXT")


def parse_table(lines, start_idx):
    """Parse a markdown table starting at start_idx. Returns (rows, end_idx)."""
    rows = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        # Skip separator rows like |---|---|
        if re.match(r"^\|[\s\-:|]+\|$", line):
            i += 1
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        rows.append(cells)
        i += 1
    return rows, i


def clean_markdown_inline(text):
    """Strip bold/italic markdown markers and links for plain-text rendering."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text.strip()


def render_table(pdf, rows):
    """Render a table with header styling."""
    if not rows:
        return

    num_cols = len(rows[0])
    usable = 190
    col_w = usable / num_cols

    # Compute column widths proportional to content
    max_lens = [0] * num_cols
    for row in rows:
        for j, cell in enumerate(row):
            max_lens[j] = max(max_lens[j], len(clean_markdown_inline(cell)))
    total_len = sum(max_lens) or 1
    col_widths = [max(20, usable * (ml / total_len)) for ml in max_lens]
    # Normalize
    s = sum(col_widths)
    col_widths = [w * usable / s for w in col_widths]

    for i, row in enumerate(rows):
        if i == 0:
            # Header
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(46, 134, 193)
            pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_font("Helvetica", "", 8)
            if i % 2 == 0:
                pdf.set_fill_color(248, 249, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 30, 30)

        max_h = 5
        cell_texts = []
        for j, cell in enumerate(row):
            txt = clean_markdown_inline(cell)
            cell_texts.append(txt)
            # Estimate height needed
            w = col_widths[j] if j < len(col_widths) else col_w
            n_lines = max(1, len(txt) * pdf.get_string_width("x") / (w - 2) + 1)
            max_h = max(max_h, int(n_lines) * 4.5 + 2)

        max_h = min(max_h, 30)

        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # Check page break
        if y_start + max_h > 277:
            pdf.add_page()
            y_start = pdf.get_y()
            x_start = pdf.get_x()

        for j, txt in enumerate(cell_texts):
            w = col_widths[j] if j < len(col_widths) else col_w
            pdf.set_xy(x_start + sum(col_widths[:j]), y_start)
            fill = True
            pdf.multi_cell(w, max_h / max(1, len(txt.split("\n"))),
                           txt, border=0, fill=fill, align="L",
                           new_x="RIGHT", new_y="TOP", max_line_height=4.5)

        pdf.set_xy(x_start, y_start + max_h)

    # Reset
    pdf.set_text_color(30, 30, 30)
    pdf.ln(3)


def build_pdf():
    pdf = PLNGuidePDF()
    pdf.cover_page()

    with open(MD_PATH, "r") as f:
        md_text = f.read()

    lines = md_text.split("\n")
    i = 0
    in_code_block = False

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                pdf.set_font("Courier", "", 8)
                pdf.set_fill_color(244, 244, 244)
                pdf.set_text_color(50, 50, 50)
                i += 1
                continue
            else:
                in_code_block = False
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.ln(3)
                i += 1
                continue

        if in_code_block:
            text = line.rstrip()
            if pdf.get_y() > 270:
                pdf.add_page()
            pdf.cell(190, 4.5, "  " + text, fill=True, new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            pdf.ln(2)
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            pdf.ln(3)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            i += 1
            continue

        # Table of Contents links (skip anchor links)
        if re.match(r"^\d+\.\s+\[", stripped):
            text = clean_markdown_inline(stripped)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(46, 134, 193)
            pdf.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        # Headings
        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(26, 82, 118)
            pdf.cell(0, 12, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(46, 134, 193)
            pdf.set_line_width(0.6)
            pdf.line(10, pdf.get_y() + 1, 200, pdf.get_y() + 1)
            pdf.ln(6)
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        if stripped.startswith("## "):
            pdf.ln(5)
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(26, 82, 118)
            pdf.cell(0, 10, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(213, 216, 220)
            pdf.set_line_width(0.3)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        if stripped.startswith("### "):
            pdf.ln(3)
            if pdf.get_y() > 260:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(46, 64, 83)
            pdf.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        if stripped.startswith("#### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(46, 134, 193)
            pdf.cell(0, 7, stripped[5:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            text = clean_markdown_inline(stripped[2:])
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(26, 82, 118)
            pdf.set_fill_color(234, 242, 248)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_draw_color(46, 134, 193)
            pdf.set_line_width(0.8)
            pdf.line(x, y, x, y + 8)
            pdf.set_x(x + 4)
            pdf.multi_cell(180, 5.5, text, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_text_color(30, 30, 30)
            i += 1
            continue

        # Table
        if stripped.startswith("|"):
            rows, end_idx = parse_table(lines, i)
            if rows:
                render_table(pdf, rows)
            i = end_idx
            continue

        # Bold line (standalone)
        if stripped.startswith("**") and stripped.endswith("**"):
            text = stripped[2:-2]
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(190, 5.5, text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            i += 1
            continue

        # Bullet points - use dash instead of Unicode bullet for Helvetica compatibility
        if stripped.startswith("- ") or stripped.startswith("* "):
            text = clean_markdown_inline(stripped[2:])
            pdf.set_font("Helvetica", "", 9.5)
            x = pdf.get_x()
            if pdf.get_y() > 272:
                pdf.add_page()
            pdf.cell(5, 5, "-", new_x="RIGHT", new_y="TOP")
            pdf.multi_cell(180, 5, text, new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if m:
            num = m.group(1)
            text = clean_markdown_inline(m.group(2))
            pdf.set_font("Helvetica", "B", 9.5)
            if pdf.get_y() > 272:
                pdf.add_page()
            pdf.cell(8, 5, f"{num}.", new_x="RIGHT", new_y="TOP")
            pdf.set_font("Helvetica", "", 9.5)
            pdf.multi_cell(178, 5, text, new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Regular paragraph
        text = clean_markdown_inline(stripped)
        pdf.set_font("Helvetica", "", 10)
        if pdf.get_y() > 272:
            pdf.add_page()
        pdf.multi_cell(190, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        i += 1

    pdf.output(PDF_PATH)
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"PDF saved: {PDF_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build_pdf()
