from math import atan2, cos, sin
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 582, 579
FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
REGULAR, BOLD = "Helvetica", "Helvetica-Bold"
regular_font = FONT_DIR / "LiberationSans-Regular.ttf"
bold_font = FONT_DIR / "LiberationSans-Bold.ttf"
if regular_font.is_file() and bold_font.is_file():
    pdfmetrics.registerFont(TTFont("Diagram", str(regular_font)))
    pdfmetrics.registerFont(TTFont("DiagramBold", str(bold_font)))
    REGULAR, BOLD = "Diagram", "DiagramBold"
c = Canvas(str(ROOT / "Paper4_Analytical_Framework.pdf"), pagesize=(WIDTH, HEIGHT))
c.setTitle("Paper 4: Overall analytical framework")

# Palette: Okabe-Ito hues as light tints with a saturated outline of the same hue.
# Colour-blind safe, and the four classes also differ in lightness (greyscale print).
INK = "#111111"      # titles
BODY = "#2B2B2B"     # body text
LINE = "#3A3A3A"     # connectors
INPUT = ("#F6F6F6", "#7A7A7A")        # exogenous inputs (Ariadne)      - grey
SIM = ("#C6DEF1", "#0072B2")          # ASSUME simulation stages 1-2    - blue
DECISION = ("#FCE8BC", "#B97A00")     # investment / adoption / demand  - orange
ASSESSMENT = ("#AEDCC6", "#007A57")   # outcome assessment              - bluish green


def label(x, y, content, size=10.4, *, bold=False, color=BODY):
    font = BOLD if bold else REGULAR
    c.setFont(font, size)
    c.setFillColor(HexColor(color))
    c.drawCentredString(x, HEIGHT - y, content)


def box(x, y, w, h, title, lines, *, style=DECISION):
    fill, stroke = style
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(1.0)
    c.roundRect(x, HEIGHT - y - h, w, h, 5, fill=1, stroke=1)
    tx = x + w / 2
    first = y + (h - (14.2 * len(lines))) / 2 + 3.2
    label(tx, first, title, 13.2, bold=True, color=INK)
    for i, line in enumerate(lines, 1):
        label(tx, first + 14.2 * i, line)


def connector(points, *, arrow=True, dashed=False):
    colour = HexColor(LINE)
    c.setStrokeColor(colour)
    c.setFillColor(colour)
    c.setLineWidth(1.1)
    if dashed:
        c.setDash(3.3, 2.6)
    p = c.beginPath()
    p.moveTo(points[0][0], HEIGHT - points[0][1])
    for x, y in points[1:]:
        p.lineTo(x, HEIGHT - y)
    c.drawPath(p)
    c.setDash()
    if arrow:
        x, y = points[-1]
        px, py = points[-2]
        direction = atan2(y - py, x - px)
        head = c.beginPath()
        head.moveTo(x, HEIGHT - y)
        for side in (-0.49, 0.49):
            vx = x - 7.2 * cos(direction + side)
            vy = y - 7.2 * sin(direction + side)
            head.lineTo(vx, HEIGHT - vy)
        head.close()
        c.drawPath(head, stroke=0, fill=1)


# Inputs enter at the stage where they are used.
box(12, 10, 558, 48, "Six Ariadne transition pathways",
    ["Power-system data, energy-carrier prices and sectoral demand"], style=INPUT)
connector([(375, 58), (375, 79)])
box(180, 79, 390, 55, "1. ASSUME | Initial market simulation",
    ["Power-system data for each pathway", "Day-ahead and negative aFRR price signals"], style=SIM)

connector([(272, 134), (272, 163)])
connector([(477, 134), (477, 145), (374, 145), (374, 249), (477, 249), (477, 268)])
label(418, 159, "Day-ahead prices", 9.1, color=LINE)
box(12, 163, 150, 76, "Ariadne prices",
    ["Other energy-carrier", "prices for industry"], style=INPUT)
connector([(162, 201), (180, 201)])
box(180, 163, 184, 76, "2. ASSUME | Plant dispatch",
    ["Industrial demand-side agents", "Day-ahead and negative aFRR", "prices plus other carrier prices"], style=SIM)
box(386, 170, 184, 64, "Ariadne transport inputs",
    ["Aviation and maritime demand", "Other energy-carrier prices"], style=INPUT)
connector([(529, 234), (529, 268)])
connector([(272, 239), (272, 268)])
label(319, 255, "Dispatch and costs", 9.1, color=LINE)

box(12, 268, 150, 76, "Ariadne production",
    ["Primary steel and", "clinker demand"], style=INPUT)
connector([(162, 306), (180, 306)])
box(180, 268, 184, 76, "Industrial investment",
    ["Primary steel and clinker", "Technology choices under", "belief and risk assumptions"])
box(386, 268, 184, 76, "Technology diffusion",
    ["Aviation and maritime transport", "Fleet technology choices under", "belief and risk assumptions"])
connector([(272, 344), (272, 360), (375, 360)], arrow=False)
connector([(478, 344), (478, 360), (375, 360)], arrow=False)
connector([(375, 360), (375, 378)])

box(180, 378, 390, 65, "Sectoral energy demand and flexibility",
    ["Electricity demand by location and time",
     "Direct use and domestic hydrogen and synthetic-fuel production",
     "Industrial flexibility"])
connector([(375, 443), (375, 461)])
box(180, 461, 390, 50, "3. ASSUME | Electricity-market assessment",
    ["Day-ahead and aFRR markets with sectoral loads and plant flexibility",
     "Evaluation under the conditions of each assessment pathway"], style=ASSESSMENT)
connector([(375, 511), (375, 532)])
label(430, 525, "Market schedules", 9.1, color=LINE)
box(180, 532, 390, 38, "Transmission-network assessment",
    ["Power flows, congestion and redispatch"], style=ASSESSMENT)

# Legend (colour classes), bottom-left
lx, ly = 12, 400
label(lx + 75, ly, "Legend", 10.4, bold=True, color=INK)
for i, (name, st) in enumerate([("Exogenous Ariadne input", INPUT),
                                ("ASSUME simulation", SIM),
                                ("Decision and demand model", DECISION),
                                ("Outcome assessment", ASSESSMENT)]):
    y = ly + 14 + i * 24
    c.setFillColor(HexColor(st[0])); c.setStrokeColor(HexColor(st[1])); c.setLineWidth(1.0)
    c.roundRect(lx, HEIGHT - y - 14, 22, 14, 3, fill=1, stroke=1)
    c.setFont(REGULAR, 9.1); c.setFillColor(HexColor(BODY))
    c.drawString(lx + 30, HEIGHT - y - 10.5, name)

c.showPage()
c.save()
