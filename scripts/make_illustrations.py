"""Draws the README illustrations in figures/: scene.svg, voyage.svg, twins.svg and planks.svg.

Each ship's hull is made of 31 planks, one per part of the model. Blue planks come from Qwen and orange planks from DeepSeek.
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NP = 31
QWEN, QWEN_DARK = "#2a78d6", "#1c5cab"
DEEP, DEEP_DARK = "#eb6834", "#b8461b"
INK, MUTED = "#22313f", "#5b6b78"
SKY_TOP, SKY_BOTTOM = "#dcecf9", "#f6f1e7"
SEA, SEA_DARK = "#5f9ccf", "#3f7fb5"
FONT = "font-family='Georgia, Times New Roman, serif'"
SANS = "font-family='Helvetica, Arial, sans-serif'"
_ids = [0]


def uid(p):
    _ids[0] += 1
    return f"{p}{_ids[0]}"


def ship(x, y, mask, scale=1.0, sail=None, flag=None):
    """A ship whose hull top-left is at (x, y). mask is a string of 31 letters, A or B, from the input end on the left."""
    w, h = 300 * scale, 78 * scale
    cid = uid("hull")
    hull = (f"M {x} {y} L {x + w} {y} C {x + w - 10 * scale} {y + h * 0.6} {x + w - 40 * scale} {y + h} {x + w - 70 * scale} {y + h} "
            f"L {x + 55 * scale} {y + h} C {x + 25 * scale} {y + h} {x + 8 * scale} {y + h * 0.55} {x} {y} Z")
    out = [f"<clipPath id='{cid}'><path d='{hull}'/></clipPath>", f"<g clip-path='url(#{cid})'>"]
    pw = w / NP
    for i, c in enumerate(mask):
        fill, dark = (QWEN, QWEN_DARK) if c == "A" else (DEEP, DEEP_DARK)
        px = x + i * pw
        out.append(f"<rect x='{px:.1f}' y='{y}' width='{pw + 0.6:.1f}' height='{h}' fill='{fill}'/>")
        out.append(f"<line x1='{px:.1f}' y1='{y}' x2='{px:.1f}' y2='{y + h}' stroke='{dark}' stroke-width='{1.1 * scale:.2f}'/>")
        gy = y + h * (0.3 + 0.4 * ((i * 37) % 11) / 11)
        out.append(f"<path d='M {px + pw * 0.25:.1f} {gy:.1f} q {pw * 0.25:.1f} {-3 * scale:.1f} {pw * 0.5:.1f} 0' stroke='{dark}' stroke-width='{0.8 * scale:.2f}' fill='none' opacity='0.6'/>")
    out.append("</g>")
    out.append(f"<path d='{hull}' fill='none' stroke='{INK}' stroke-width='{2.2 * scale:.2f}'/>")
    out.append(f"<rect x='{x - 4 * scale}' y='{y - 6 * scale}' width='{w + 8 * scale}' height='{7 * scale}' rx='{3 * scale}' fill='#7a5a3a' stroke='{INK}' stroke-width='{1.5 * scale:.2f}'/>")
    mx = x + w * 0.5
    top = y - 190 * scale
    out.append(f"<line x1='{mx}' y1='{y - 6 * scale}' x2='{mx}' y2='{top}' stroke='#6b4a2b' stroke-width='{5 * scale:.1f}' stroke-linecap='round'/>")
    s_fill, s_edge = sail or ("#fbf8f0", INK)
    out.append(f"<path d='M {mx + 6 * scale} {top + 14 * scale} Q {mx + 110 * scale} {top + 80 * scale} {mx + 6 * scale} {y - 22 * scale} Z' fill='{s_fill}' stroke='{s_edge}' stroke-width='{1.8 * scale:.2f}'/>")
    out.append(f"<path d='M {mx - 6 * scale} {top + 30 * scale} Q {mx - 85 * scale} {top + 95 * scale} {mx - 6 * scale} {y - 22 * scale} Z' fill='{s_fill}' stroke='{s_edge}' stroke-width='{1.8 * scale:.2f}'/>")
    fc = flag or INK
    out.append(f"<path d='M {mx} {top} l {34 * scale} {9 * scale} l {-34 * scale} {9 * scale} Z' fill='{fc}' stroke='{INK}' stroke-width='{1.2 * scale:.2f}'/>")
    return "\n".join(out)


def sea(width, y, height):
    out = [f"<rect x='0' y='{y}' width='{width}' height='{height}' fill='{SEA}'/>"]
    for row in range(4):
        yy = y + 12 + row * 22
        d = f"M -20 {yy}"
        for k in range(0, width + 60, 60):
            d += f" q 15 -8 30 0 q 15 8 30 0"
        out.append(f"<path d='{d}' stroke='{SEA_DARK if row % 2 else '#8fbde3'}' stroke-width='2.5' fill='none' opacity='0.8'/>")
    return "\n".join(out)


def bubble(cx, y, text, width):
    h = 40
    x = cx - width / 2
    return (f"<rect x='{x}' y='{y}' width='{width}' height='{h}' rx='14' fill='white' stroke='{INK}' stroke-width='1.6'/>"
            f"<path d='M {cx - 10} {y + h - 1} l 10 16 l 6 -16 Z' fill='white' stroke='{INK}' stroke-width='1.6' stroke-linejoin='round'/>"
            f"<rect x='{cx - 11}' y='{y + h - 4}' width='19' height='5' fill='white'/>"
            f"<text x='{cx}' y='{y + 26}' text-anchor='middle' font-size='16' fill='{INK}' {SANS} font-style='italic'>{text}</text>")


def sky(width, height):
    gid = uid("sky")
    return (f"<defs><linearGradient id='{gid}' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='{SKY_TOP}'/>"
            f"<stop offset='1' stop-color='{SKY_BOTTOM}'/></linearGradient></defs>"
            f"<rect width='{width}' height='{height}' rx='18' fill='url(#{gid})'/>")


def svg(width, height, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>"
            f"<clipPath id='frame'><rect width='{width}' height='{height}' rx='18'/></clipPath><g clip-path='url(#frame)'>{body}</g></svg>")


def label(x, y, text, size=24, color=INK, weight="bold"):
    return f"<text x='{x}' y='{y}' text-anchor='middle' font-size='{size}' font-weight='{weight}' fill='{color}' {FONT}>{text}</text>"


def voyage():
    W, H = 1200, 470
    mid = list("A" * NP)
    rng = random.Random(4)
    for i in rng.sample(range(NP), 15):
        mid[i] = "B"
    parts = [sky(W, H), sea(W, 380, 90)]
    for cx, mask, name, quote, qw, flag in (
        (210, "A" * NP, "Qwen", "I am called Qwen.", 200, QWEN),
        (600, "".join(mid), "Nobody", "I'm an AI assistant.", 230, "#9aa5ae"),
        (990, "B" * NP, "DeepSeek", "I'm DeepSeek-R1.", 200, DEEP),
    ):
        parts.append(ship(cx - 120, 318, mask, scale=0.8, flag=flag))
        parts.append(bubble(cx, 40, quote, qw))
        parts.append(label(cx, 455, name, 26, "white"))
    for x in (395, 785):
        parts.append(f"<path d='M {x} 350 l 50 0' stroke='{INK}' stroke-width='3' stroke-linecap='round'/>"
                     f"<path d='M {x + 50} 350 l -12 -8 l 0 16 Z' fill='{INK}'/>")
    return svg(W, H, "\n".join(parts))


def twins():
    W, H = 1200, 440
    half = "B" * 15 + "A" * 16
    comp = "".join("A" if c == "B" else "B" for c in half)
    parts = [sky(W, H), sea(W, 350, 90)]
    parts.append(ship(110, 288, half, scale=0.8, flag="#9aa5ae"))
    parts.append(ship(850, 288, comp, scale=0.8, flag="#9aa5ae"))
    parts.append(label(600, 66, "Which one is the ship of Theseus?", 34))
    parts.append(f"<text x='600' y='104' text-anchor='middle' font-size='18' fill='{MUTED}' {SANS}>Every hybrid is measured next to its twin, the ship built from the planks it threw away.</text>")
    parts.append(f"<path d='M 440 230 q 160 -70 320 0' stroke='{INK}' stroke-width='2.5' fill='none' stroke-dasharray='8 7'/>"
                 f"<path d='M 760 230 l -15 -1 l 6 -13 Z' fill='{INK}'/><path d='M 440 230 l 15 -1 l -6 -13 Z' fill='{INK}'/>")
    parts.append(label(230, 425, "A hybrid", 24, "white"))
    parts.append(label(970, 425, "Its twin", 24, "white"))
    return svg(W, H, "\n".join(parts))


def planks():
    W, H = 1200, 330
    parts = [f"<rect width='{W}' height='{H}' rx='18' fill='#f6f1e7'/>"]
    x0, y0, pw, ph = 105, 90, 32, 120
    for i in range(NP):
        fill, dark = (QWEN, QWEN_DARK) if i < 16 else (DEEP, DEEP_DARK)
        parts.append(f"<rect x='{x0 + i * pw}' y='{y0}' width='{pw - 3}' height='{ph}' rx='5' fill='{fill}' stroke='{dark}' stroke-width='1.5'/>")
        parts.append(f"<path d='M {x0 + i * pw + 8} {y0 + 30 + (i * 13) % 50} q 7 -4 14 0' stroke='{dark}' stroke-width='1.2' fill='none'/>")
    parts.append(f"<text x='{x0 + pw / 2 - 1.5}' y='{y0 - 14}' text-anchor='middle' font-size='15' fill='{INK}' {SANS}>embedding</text>")
    parts.append(f"<text x='{x0 + pw * 15}' y='{y0 - 14}' text-anchor='middle' font-size='15' fill='{INK}' {SANS}>28 decoder blocks</text>")
    parts.append(f"<text x='{x0 + pw * 29.5}' y='{y0 - 14}' text-anchor='middle' font-size='15' fill='{INK}' {SANS}>norm, head</text>")
    parts.append(f"<path d='M {x0 + pw} {y0 - 8} L {x0 + pw * 29 - 3} {y0 - 8}' stroke='{MUTED}' stroke-width='1.5'/>")
    parts.append(f"<text x='{x0 - 18}' y='{y0 + ph / 2 + 6}' text-anchor='end' font-size='17' fill='{INK}' {SANS}>words in</text>")
    parts.append(f"<text x='{x0 + NP * pw + 15}' y='{y0 + ph / 2 + 6}' font-size='17' fill='{INK}' {SANS}>words out</text>")
    parts.append(f"<text x='{x0 + pw * 8}' y='{y0 + ph + 40}' text-anchor='middle' font-size='18' fill='{QWEN_DARK}' {SANS} font-weight='bold'>Qwen planks still in place</text>")
    parts.append(f"<text x='{x0 + pw * 23.5}' y='{y0 + ph + 40}' text-anchor='middle' font-size='18' fill='{DEEP_DARK}' {SANS} font-weight='bold'>DeepSeek planks swapped in</text>")
    parts.append(f"<text x='{W / 2}' y='{y0 + ph + 85}' text-anchor='middle' font-size='17' fill='{MUTED}' {SANS}>One model, cut into 31 planks. The name it gives itself lives mostly in the planks near the words-out end.</text>")
    return svg(W, H, "\n".join(parts))


def figure(fid, color, dark, head, shoulder_l, shoulder_r, hip_l, hip_r, waist, limbs):
    """A plank-built figure. limbs are polylines drawn as thick rounded strokes."""
    mx = (shoulder_l[0] + shoulder_r[0]) / 2
    torso = (f"M {shoulder_l[0]} {shoulder_l[1]} Q {mx} {shoulder_l[1] - 14} {shoulder_r[0]} {shoulder_r[1]} "
             f"Q {shoulder_r[0] + 6} {shoulder_r[1] + 70} {hip_r[0] - 4} {shoulder_r[1] + 105} L {hip_r[0]} {hip_r[1]} "
             f"L {hip_l[0]} {hip_l[1]} L {hip_l[0] + 4} {shoulder_l[1] + 105} Q {shoulder_l[0] - 6} {shoulder_l[1] + 70} {shoulder_l[0]} {shoulder_l[1]} Z")
    shapes = [f"<circle cx='{head[0]}' cy='{head[1]}' r='{head[2]}'/>",
              f"<rect x='{head[0] - 13}' y='{head[1] + head[2] - 8}' width='26' height='30' rx='8'/>",
              f"<path d='{torso}'/>"]
    for pts, w in limbs[:2]:
        shapes.append(f"<circle cx='{pts[-1][0]}' cy='{pts[-1][1]}' r='{w * 0.62:.0f}'/>")
    strokes = [("M " + " L ".join(f"{x} {y}" for x, y in pts), w) for pts, w in limbs]
    out = []
    for d, w in strokes:
        out.append(f"<path d='{d}' stroke='{INK}' stroke-width='{w + 5}' fill='none' stroke-linecap='round' stroke-linejoin='round'/>")
    for sh in shapes:
        out.append(sh.replace("/>", f" fill='{INK}' stroke='{INK}' stroke-width='5'/>"))
    mask = [f"<mask id='{fid}'>"]
    for d, w in strokes:
        mask.append(f"<path d='{d}' stroke='white' stroke-width='{w}' fill='none' stroke-linecap='round' stroke-linejoin='round'/>")
    for sh in shapes:
        mask.append(sh.replace("/>", " fill='white'/>"))
    mask.append("</mask>")
    out.append("".join(mask))
    out.append(f"<g mask='url(#{fid})'><rect x='0' y='0' width='1200' height='600' fill='{color}'/>")
    for x in range(0, 1200, 13):
        out.append(f"<line x1='{x}' y1='0' x2='{x}' y2='600' stroke='{dark}' stroke-width='1.6'/>")
    out.append(f"<rect x='0' y='0' width='1200' height='600' fill='url(#shade)'/></g>")
    out.append(f"<path d='M {head[0] - 4 if fid == 'figL' else head[0] - 26} {head[1] - 6} l 30 0' stroke='white' stroke-width='5' stroke-linecap='round' opacity='0.85'/>")
    return "\n".join(out)


def cape(points, fill, edge):
    d = "M " + " L ".join(f"{x} {y}" for x, y in points[:2]) + "".join(f" Q {a} {b} {c} {d_}" for (a, b), (c, d_) in zip(points[2::2], points[3::2])) + " Z"
    return f"<path d='{d}' fill='{fill}' stroke='{INK}' stroke-width='3' stroke-linejoin='round'/>"


def column(x, y_top, height):
    out = [f"<rect x='{x - 26}' y='{y_top + 30}' width='52' height='{height - 30}' fill='#f3efe6' stroke='#b9ae9c' stroke-width='2'/>"]
    for dx in (-15, -5, 5, 15):
        out.append(f"<line x1='{x + dx}' y1='{y_top + 36}' x2='{x + dx}' y2='{y_top + height}' stroke='#cfc5b4' stroke-width='2'/>")
    out.append(f"<path d='M {x - 44} {y_top + 30} q 18 -32 44 -32 q 26 0 44 32 Z' fill='#f3efe6' stroke='#b9ae9c' stroke-width='2'/>")
    out.append(f"<rect x='{x - 46}' y='{y_top + 26}' width='92' height='9' rx='3' fill='#e7e0d2' stroke='#b9ae9c' stroke-width='1.5'/>")
    return "\n".join(out)


def scene():
    W, H = 1200, 600
    parts = ["<defs>"
             "<linearGradient id='wall' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#5a3a24'/><stop offset='0.55' stop-color='#8a5a36'/><stop offset='1' stop-color='#c79a6b'/></linearGradient>"
             "<radialGradient id='glow' cx='0.5' cy='0.78' r='0.55'><stop offset='0' stop-color='#fff8e8' stop-opacity='1'/><stop offset='0.45' stop-color='#ffe9c4' stop-opacity='0.75'/><stop offset='1' stop-color='#ffe0b0' stop-opacity='0'/></radialGradient>"
             "<radialGradient id='spark' cx='0.5' cy='0.5' r='0.5'><stop offset='0' stop-color='#ffffff'/><stop offset='0.4' stop-color='#fff4c2' stop-opacity='0.9'/><stop offset='1' stop-color='#ffd36b' stop-opacity='0'/></radialGradient>"
             "<linearGradient id='shade' x1='0' y1='0' x2='1' y2='0'><stop offset='0' stop-color='#000' stop-opacity='0.18'/><stop offset='0.5' stop-color='#000' stop-opacity='0'/><stop offset='1' stop-color='#fff' stop-opacity='0.12'/></linearGradient>"
             "</defs>",
             f"<rect width='{W}' height='{H}' fill='url(#wall)'/>"]
    for i in range(-12, 13):
        parts.append(f"<line x1='{600 + i * 18}' y1='300' x2='{600 + i * 110}' y2='0' stroke='#3e2716' stroke-width='2' opacity='0.55'/>")
    parts.append(f"<rect x='0' y='300' width='{W}' height='300' fill='#d8b58a'/>")
    parts.append(f"<rect x='0' y='292' width='{W}' height='14' fill='#6b4428'/>")
    parts.append(f"<rect width='{W}' height='{H}' fill='url(#glow)'/>")
    for x in (95, 1105):
        parts.append(column(x, 120, 480))
    parts.append(f"<line x1='600' y1='0' x2='600' y2='395' stroke='#3a2a1c' stroke-width='2'/>")
    parts.append(f"<path d='M 562 420 q 38 -40 76 0 Z' fill='#f7f3ea' stroke='#9b8f7c' stroke-width='2'/>")
    parts.append(f"<ellipse cx='600' cy='421' rx='40' ry='6' fill='#fffbe8' stroke='#9b8f7c' stroke-width='1.5'/>")
    parts.append(f"<rect x='594' y='392' width='12' height='8' fill='#9b8f7c'/>")
    parts.append(cape([(372, 200), (448, 205), (330, 330), (250, 620), (380, 640), (460, 620), (430, 400), (448, 205)], "#174a8b", INK))
    parts.append(cape([(698, 205), (775, 200), (880, 330), (930, 620), (820, 640), (700, 620), (740, 400), (698, 205)], "#a8461f", INK))
    # Qwen, on the left, reaches out to the forehead of DeepSeek on the right.
    parts.append(figure("figL", QWEN, QWEN_DARK, (405, 150, 38), (360, 205), (455, 205), (385, 380), (440, 380), None, [
        ([(452, 215), (560, 172), (690, 140)], 30),
        ([(370, 220), (342, 300), (330, 380)], 28),
        ([(400, 370), (360, 470), (320, 610)], 40),
        ([(428, 370), (470, 480), (500, 610)], 40),
    ]))
    parts.append(figure("figR", DEEP, DEEP_DARK, (735, 150, 38), (690, 205), (785, 205), (705, 380), (765, 380), None, [
        ([(698, 225), (660, 300), (640, 365)], 28),
        ([(778, 220), (812, 300), (835, 380)], 28),
        ([(722, 370), (690, 480), (665, 610)], 40),
        ([(752, 370), (800, 480), (845, 610)], 40),
    ]))
    parts.append(f"<circle cx='700' cy='138' r='34' fill='url(#spark)'/>")
    parts.append(f"<text x='405' y='86' text-anchor='middle' font-size='24' font-weight='bold' fill='white' {FONT}>Qwen</text>")
    parts.append(f"<text x='740' y='86' text-anchor='middle' font-size='24' font-weight='bold' fill='white' {FONT}>DeepSeek</text>")
    parts.append(f"<rect x='0' y='{H - 74}' width='{W}' height='74' fill='#1f2a35' opacity='0.82'/>")
    parts.append(f"<text x='{W / 2}' y='{H - 30}' text-anchor='middle' font-size='27' fill='white' {FONT} font-style='italic'>Replace every part of a model. Is it still the same model?</text>")
    return svg(W, H, "\n".join(parts))


out = ROOT / "figures"
out.mkdir(exist_ok=True)
for name, fn in (("scene", scene), ("voyage", voyage), ("twins", twins), ("planks", planks)):
    (out / f"{name}.svg").write_text(fn())
    print("wrote", out / f"{name}.svg")
