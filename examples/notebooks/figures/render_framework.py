from math import atan2, cos, sin
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent
# The layout is written in pixels of a 1146 x 919 canvas; one pixel is half a point.
W_PX, H_PX = 1146, 919
SCALE = 0.5
FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
REGULAR, BOLD = "Helvetica", "Helvetica-Bold"
regular_font = FONT_DIR / "LiberationSans-Regular.ttf"
bold_font = FONT_DIR / "LiberationSans-Bold.ttf"
if regular_font.is_file() and bold_font.is_file():
    pdfmetrics.registerFont(TTFont("Diagram", str(regular_font)))
    pdfmetrics.registerFont(TTFont("DiagramBold", str(bold_font)))
    REGULAR, BOLD = "Diagram", "DiagramBold"
c = Canvas(str(ROOT / "Paper4_Analytical_Framework.pdf"), pagesize=(W_PX * SCALE, H_PX * SCALE))
c.setTitle("Paper 4: Overall analytical framework")
c.scale(SCALE, SCALE)

# Palette: one light tint per stage with a darker outline of the same hue, and a dark
# title colour of that hue. Stage hues stay distinguishable in greyscale by lightness.
INK = "#172B38"       # input-box titles, questions
BODY = "#314955"      # body text
BODY_SOFT = "#49616F"  # explanatory text in the comparison box
SLATE = "#365263"     # connectors into the appraisal stage
INPUT = dict(fill="#F4F6F7", stroke="#B6C4CB", inner="#B6C4CB", title=INK)           # exogenous inputs - grey
APPRAISAL = dict(fill="#EAF2F8", stroke="#9EB7C8", inner="#B0C5D3", title="#214D70")   # appraisal - blue
DEMAND = dict(fill="#EAF4EF", stroke="#91B3A2", inner="#A8C4B6", title="#265A48")      # demand - green
ASSESSMENT = dict(fill="#F2EDF6", stroke="#B3A0C1", inner="#BBA9C8", title="#59416D")  # assessment - purple
PURPLE_LINE = "#6F547F"

TOP_TITLE, MAIN_TITLE, INNER_TITLE, BODY_SIZE, LIST_SIZE, SMALL = 22.0, 25.0, 22.0, 20.35, 21.4, 18.0
RIGHT_INNER_TITLE, QUESTION_SIZE, NOTE_SIZE = 22.8, 22.3, 20.2
CAP = 0.725  # cap/ascender height as a fraction of the font size (Liberation Sans)


def Y(y):
    return H_PX - y


def label(x, top, content, size=BODY_SIZE, *, bold=False, color=BODY):
    """Centre one line of text at x; `top` is the top of its capitals in pixels."""
    c.setFont(BOLD if bold else REGULAR, size)
    c.setFillColor(HexColor(color))
    c.drawCentredString(x, Y(top + CAP * size), content)


def lines(x, top, items, pitch, *, size=BODY_SIZE, bold=False, color=BODY):
    for i, text in enumerate(items):
        label(x, top + pitch * i, text, size, bold=bold, color=color)


def rect(x0, y0, x1, y1, fill, stroke, radius, width=1.0):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(width * 2)  # width given in output points
    c.roundRect(x0, Y(y1), x1 - x0, y1 - y0, radius, fill=1, stroke=1)


def stage(x0, y0, x1, y1, style):
    rect(x0, y0, x1, y1, style["fill"], style["stroke"], 13)


def inner(x0, y0, x1, y1, style):
    rect(x0, y0, x1, y1, "#FFFFFF", style["inner"], 9, width=0.9)


def connector(points, *, color=SLATE, arrow=True, dashed=False):
    colour = HexColor(color)
    c.setStrokeColor(colour)
    c.setFillColor(colour)
    c.setLineWidth(2.2)
    if dashed:
        c.setDash(6.6, 5.2)
    p = c.beginPath()
    p.moveTo(points[0][0], Y(points[0][1]))
    for x, y in points[1:]:
        p.lineTo(x, Y(y))
    c.drawPath(p)
    c.setDash()
    if arrow:
        x, y = points[-1]
        px, py = points[-2]
        direction = atan2(y - py, x - px)
        head = c.beginPath()
        head.moveTo(x, Y(y))
        for side in (-0.49, 0.49):
            vx = x - 14.4 * cos(direction + side)
            vy = y - 14.4 * sin(direction + side)
            head.lineTo(vx, Y(vy))
        head.close()
        c.drawPath(head, stroke=0, fill=1)


# ---------------------------------------------------------------- inputs (top row)
for x0, title, body in [
    (18, ["Uncertainty about", "future developments"], ["Technology costs, energy prices,", "activity, policy and infrastructure"]),
    (409, ["Baseline market simulations"], ["Electricity price signals", "by pathway and year"]),
    (800, ["Assessment pathway and year"], ["Industrial operating conditions,", "refinery capacity, supply and grid"]),
]:
    stage(x0, 11, x0 + 336, 143, INPUT)
    cx = x0 + 168
    lines(cx, 27 if len(title) == 2 else 40, title, 25, size=TOP_TITLE, bold=True, color=INK)
    lines(cx, 77, body, 25)

# ---------------------------------------------------------------- connectors into the stages
connector([(186, 144), (186, 233)])
connector([(577, 144), (577, 190), (276, 190), (276, 233)])
connector([(968, 144), (968, 233)], color=PURPLE_LINE)
connector([(968, 190), (627, 190), (627, 233)], color=PURPLE_LINE)
label(797, 166, "Industrial dispatch and nodal allocation", SMALL, color=PURPLE_LINE)

# ---------------------------------------------------------------- stage 1: investment appraisal
stage(18, 234, 354, 668, APPRAISAL)
lines(186, 258, ["Investment appraisal", "and technology adoption"], 30, size=MAIN_TITLE, bold=True, color=APPRAISAL["title"])
inner(35, 331, 337, 404, APPRAISAL)
lines(186, 343, ["Pathway-specific payoffs", "Beliefs and decision criteria"], 25, size=21.75, bold=True, color=APPRAISAL["title"])
lines(186, 424, ["Aviation", "Maritime transport", "Primary steel production", "Clinker production"], 31.6, size=LIST_SIZE)
lines(186, 584, ["Technologies, fuel choices", "and timing of adoption"], 26, size=LIST_SIZE - 0.2, bold=True, color=APPRAISAL["title"])

# ---------------------------------------------------------------- stage 2: electricity demand and flexibility
stage(409, 234, 745, 668, DEMAND)
label(577, 273, "Electricity demand", MAIN_TITLE, bold=True, color=DEMAND["title"])
for y0, name, text in [
    (331, "Volume", "Annual electricity demand"),
    (410, "Timing", "When electricity is needed"),
    (489, "Location", "Allocation to grid nodes"),
    (568, "Industrial flexibility", "Negative aFRR energy bids"),
]:
    inner(426, y0, 728, y0 + 70, DEMAND)
    label(577, y0 + 12, name, INNER_TITLE - 0.5, bold=True, color=DEMAND["title"])
    label(577, y0 + 38, text, BODY_SIZE)

# ---------------------------------------------------------------- stage 3: market and transmission assessment
stage(800, 234, 1136, 668, ASSESSMENT)
lines(968, 258, ["Electricity-market and", "transmission assessment"], 30, size=MAIN_TITLE, bold=True, color=ASSESSMENT["title"])
inner(818, 331, 1120, 442, ASSESSMENT)
label(969, 349, "Market clearing", RIGHT_INNER_TITLE, bold=True, color=ASSESSMENT["title"])
lines(969, 381, ["Day-ahead and aFRR prices", "Cleared demand and dispatch"], 25)
connector([(968, 442), (968, 486)], color=PURPLE_LINE)
inner(818, 486, 1120, 659, ASSESSMENT)
label(969, 502, "Transmission analysis", RIGHT_INNER_TITLE, bold=True, color=ASSESSMENT["title"])
lines(969, 536, ["Line loading and congestion", "Renewable curtailment", "Upward redispatch volumes", "and payments"], 25.6)

# ---------------------------------------------------------------- flow between the stages
connector([(358, 447), (406, 447)])
connector([(749, 447), (797, 447)])

# ---------------------------------------------------------------- comparison of outcomes
connector([(968, 669), (968, 694.5), (577, 694.5), (577, 722)], color=PURPLE_LINE)
stage(18, 723, 1136, 916, ASSESSMENT)
label(577, 745, "Comparison of market and transmission outcomes", MAIN_TITLE, bold=True, color=ASSESSMENT["title"])
c.setStrokeColor(HexColor("#C4B5D0"))
c.setLineWidth(2.0)
c.line(40, Y(779.5), 1116, Y(779.5))
c.line(577, Y(783), 577, Y(898))
for cx, q, b in [
    (297, ["How do outcomes differ between actors", "with different beliefs or decision criteria?"],
     ["Assess the resulting investments under", "the same pathway for the same year."]),
    (857, ["How do outcomes differ across", "future pathways?"],
     ["Assess the same investment choices under", "different pathways for the same year."]),
]:
    lines(cx, 792, q, 27, size=QUESTION_SIZE, bold=True, color=INK)
    lines(cx, 854, b, 25, size=NOTE_SIZE, color=BODY_SOFT)

c.showPage()
c.save()
