# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andreas A. Buchheit

"""Draw the GZL logo as SVG.

The mark is a cycle graph in silver: nine nodes on a circle, joined by
straight edges.  Two nodes, the terminals s (lower left) and t (upper
right), are joined by a red ring of light, the hopping.  The ring is a
circle in 3D whose diameter is the chord st: it pierces the graph plane
at s and t, so one half passes in front of the graph and the other
behind it.  Three cues carry the depth: the ring is thicker and brighter
in front, the back half is cut where a silver edge or node lies over it,
and at s and t it threads through the open node like a chain link.  The
terminals shine with the ring's light.

With an odd number of nodes no two are opposite, so the chord misses
the centre, and at s and t the ring leans out of the circle on the side
where the chord runs close to it.  That side must be the back half,
which then tucks behind the edges; the vertex sits at the top for this.

The lettering is Montserrat (SIL Open Font License 1.1), converted to
outlines, so the files need no font.

Run from the repo root:

    python3 docs/logo/make_logo.py

Writes the README logo, ``gzl-logo-wide-light.svg`` and
``gzl-logo-wide-dark.svg``, to ``docs/logo/``; ``--all`` also writes the
logo stacked over GZL, next to GZL, the mark alone and an app icon.
Needs fontTools and Montserrat; the font is found with ``kpsewhich``
(TeX Live ships it), or pass ``--font-dir``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
TITLE = 'GZL, graph zeta library'
NAME = 'Graph Zeta Library'
W = 520  # width of the drawing frame in which the mark is built


@dataclass(frozen=True)
class Style:
    n: int = 9                 # nodes on the circle
    rot: float = 180.0         # 0: a vertex at the bottom, 180: at the top
    R: float = 190.0           # circle radius
    s_angle: float = 210.0     # polar angle nearest to terminal s
    t_angle: float = 50.0      # polar angle nearest to terminal t
    node_r: float = 18.0       # node radius (centre line of the annulus)
    node_w: float = 9.5        # annulus stroke width
    edge_w: float = 9.5
    nodes: str = 'open'        # open | filled (spheres) | mixed (spheres, open terminals)
    aspect: float = 0.36       # minor / major semi-axis of the ring
    front: str = 'lower'       # half of the ring in front of the graph
    tube: float = 1.0          # ring thickness scale
    ring: str = 'red'          # red | blue | bluegreen
    theme: str = 'dark'        # dark | light: the background it sits on
    layout: str = 'stacked'    # stacked | horizontal | wide | mark | icon
    weight: str = 'Medium'     # Montserrat weight of the lettering
    cap: float = 84.0          # cap height of the lettering
    tracking: float = 0.28     # letter spacing, em
    word_gap: float = 58.0     # mark to lettering, stacked layout
    name_weight: str = 'Medium'   # the name under GZL, wide layout
    name_tracking: float = 0.06
    name_gap: float = 0.3      # GZL baseline to name, in GZL cap heights
    gap: float = 3.5           # clearance where the back half passes under
    shine: float = 1.0         # halo of ring light around the terminals
    glint: float = 0.0         # four-point glint on the terminals


RING = {
    ('red', 'dark'): dict(core='#FFF4F5', tube='#FFA3AE', mid='#FF3D5A', glow='#E8102F'),
    ('red', 'light'): dict(core='#FFB3BF', tube='#E3163A', mid='#C8102E', glow='#F0284A'),
    ('blue', 'dark'): dict(core='#F4FAFF', tube='#9BD0FF', mid='#3C87FF', glow='#1A5CFF'),
    ('blue', 'light'): dict(core='#8DC1FF', tube='#1F66F5', mid='#1A55E6', glow='#3B78FF'),
    ('bluegreen', 'dark'): dict(core='#F2FFFB', tube='#93F2D8', mid='#1CCFA3', glow='#1A5CFF'),
    ('bluegreen', 'light'): dict(core='#A8F5DD', tube='#0FAE87', mid='#0A8F70', glow='#2B6BFF'),
}

THEME = {
    'dark': dict(
        edge=[(0, '#FFFFFF'), (0.45, '#D3D9E0'), (0.75, '#A3ADB9'), (1, '#8A94A1')],
        ring=[(0, '#FFFFFF'), (0.42, '#E1E6EC'), (0.5, '#AEB7C2'), (0.78, '#8B95A2'), (1, '#C9D0D8')],
        sphere=[(0, '#FFFFFF'), (0.3, '#E3E7EC'), (0.7, '#9AA4B0'), (0.92, '#6F7985'),
                (1, '#AEB6C0')],
        word=[(0, '#FFFFFF'), (0.55, '#D5DBE2'), (1, '#9DA7B3')],
        name='#B9C1CB',
        glow=1.0, core=1.0),
    'light': dict(
        edge=[(0, '#B3BCC7'), (0.45, '#838E9C'), (0.75, '#626C7A'), (1, '#525B68')],
        ring=[(0, '#C9D0D9'), (0.42, '#9AA4B1'), (0.5, '#6A7483'), (0.78, '#4E5765'), (1, '#8A94A1')],
        sphere=[(0, '#FFFFFF'), (0.3, '#D4D9DF'), (0.7, '#7F8996'), (0.92, '#4F5866'),
                (1, '#7B8592')],
        word=[(0, '#3B4452'), (1, '#1E252F')],
        name='#4A5462',
        glow=0.42, core=0.42),
}


def f2(x):
    s = f'{x:.2f}'.rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


def stops(lst):
    return ''.join(f'<stop offset="{off}" stop-color="{col}"/>' for off, col in lst)


def ellipse(cx, cy, rx, ry, u):
    """Full ellipse, major semi-axis rx along the unit vector u."""
    phi = math.degrees(math.atan2(u[1], u[0]))
    x0, y0 = cx - rx * u[0], cy - rx * u[1]
    x1, y1 = cx + rx * u[0], cy + rx * u[1]
    return (f'M{f2(x0)} {f2(y0)}A{f2(rx)} {f2(ry)} {f2(phi)} 1 0 {f2(x1)} {f2(y1)}'
            f'A{f2(rx)} {f2(ry)} {f2(phi)} 1 0 {f2(x0)} {f2(y0)}Z')


def band(m, a, b, wf, wb, ws, u, f):
    """The ring as the even-odd region between two ellipses: wf thick at
    the front apex, wb at the back apex, ws at the terminals.  u runs
    along the chord, f points to the front half.  Shifting the two
    centres apart by (wf - wb)/2 along f is what makes it taper."""
    bo, bi = b + (wf + wb) / 4, b - (wf + wb) / 4
    sh = (wf - wb) / 4
    return (ellipse(m[0] + sh * f[0], m[1] + sh * f[1], a + ws / 2, bo, u)
            + ellipse(m[0] - sh * f[0], m[1] - sh * f[1], a - ws / 2, bi, u))


def fgrad(gid, m, f, half, fn, nstops=48):
    """Grey gradient along f through m, sampling fn(xi) at the signed
    distance xi from the chord, xi in [-half, half]; used as a mask."""
    ss = []
    for i in range(nstops + 1):
        u = i / nstops
        g = max(0, min(255, round(255 * fn(half * (1 - 2 * u)))))
        ss.append(f'<stop offset="{u:.4f}" stop-color="#{g:02x}{g:02x}{g:02x}"/>')
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
            f'x1="{f2(m[0] + half * f[0])}" y1="{f2(m[1] + half * f[1])}" '
            f'x2="{f2(m[0] - half * f[0])}" y2="{f2(m[1] - half * f[1])}">{"".join(ss)}</linearGradient>')


def geometry(st):
    """Node positions and the indices of the terminals s and t."""
    cx = 260.0
    cy = 40 + st.R + st.node_r + 8
    pts = []
    for k in range(st.n):
        th = -90 + st.rot + 360 * k / st.n
        r = math.radians(th)
        pts.append((cx + st.R * math.cos(r), cy - st.R * math.sin(r), th % 360))
    ang = lambda k, target: abs(((pts[k][2] - target + 180) % 360) - 180)
    t = min(range(st.n), key=lambda k: ang(k, st.t_angle))
    s = min(range(st.n), key=lambda k: ang(k, st.s_angle))
    return cx, cy, pts, s, t


class Lettering:
    """Text in one Montserrat weight, kerned, as outlines."""

    def __init__(self, font_dir: Path, weight: str):
        font = TTFont(font_dir / f'Montserrat-{weight}.otf')
        self.gs = font.getGlyphSet()
        self.cmap = font.getBestCmap()
        self.upm = font['head'].unitsPerEm
        self.cap_height = font['OS/2'].sCapHeight
        gpos = font['GPOS'].table
        index = sorted({i for fr in gpos.FeatureList.FeatureRecord if fr.FeatureTag == 'kern'
                        for i in fr.Feature.LookupListIndex})
        self.kern_lookups = []
        for i in index:
            lookup = gpos.LookupList.Lookup[i]
            tables = [t.ExtSubTable if lookup.LookupType == 9 else t for t in lookup.SubTable]
            self.kern_lookups.append([t for t in tables if type(t).__name__ == 'PairPos'])

    def kern(self, a, b):
        """Pair kerning of glyphs a, b in font units: each kern lookup adds
        the value of its first subtable that matches the pair."""
        total = 0
        for tables in self.kern_lookups:
            for t in tables:
                cov = t.Coverage.glyphs
                if a not in cov:
                    continue
                if t.Format == 1:
                    rec = next((r for r in t.PairSet[cov.index(a)].PairValueRecord
                                if r.SecondGlyph == b), None)
                    if rec is None:
                        continue
                    v = rec.Value1
                else:
                    c1 = t.ClassDef1.classDefs.get(a, 0)
                    c2 = t.ClassDef2.classDefs.get(b, 0)
                    v = t.Class1Record[c1].Class2Record[c2].Value1
                total += (getattr(v, 'XAdvance', 0) or 0) if v is not None else 0
                break
        return total

    def _layout(self, text, cap, tracking):
        sc = cap / self.cap_height
        names = [self.cmap[ord(ch)] for ch in text]
        x, placed = 0.0, []
        for i, name in enumerate(names):
            placed.append((name, x))
            x += self.gs[name].width + tracking * self.upm
            if i + 1 < len(names):
                x += self.kern(name, names[i + 1])
        xmin, xmax = 1e9, -1e9
        for name, ox in placed:
            bp = BoundsPen(self.gs)
            self.gs[name].draw(bp)
            if bp.bounds:
                xmin, xmax = min(xmin, ox + bp.bounds[0]), max(xmax, ox + bp.bounds[2])
        return sc, placed, xmin, xmax

    def path(self, text, cap, tracking, x, top, align='center'):
        """Outline path; x is the centre (align='center') or left ink edge."""
        sc, placed, xmin, xmax = self._layout(text, cap, tracking)
        x0 = (x - (xmax - xmin) * sc / 2 if align == 'center' else x) - xmin * sc
        base = top + cap
        out = []
        for name, ox in placed:
            pen = SVGPathPen(self.gs, ntos=f2)
            self.gs[name].draw(TransformPen(pen, (sc, 0, 0, -sc, x0 + ox * sc, base)))
            out.append(pen.getCommands())
        return ''.join(out)

    def ink_width(self, text, cap, tracking):
        sc, _, xmin, xmax = self._layout(text, cap, tracking)
        return (xmax - xmin) * sc


class Fonts:
    """One Lettering per Montserrat weight, loaded on first use."""

    def __init__(self, font_dir: Path):
        self.font_dir, self.cache = font_dir, {}

    def __call__(self, weight: str) -> Lettering:
        if weight not in self.cache:
            self.cache[weight] = Lettering(self.font_dir, weight)
        return self.cache[weight]


def build(st: Style, fonts: Fonts) -> str:
    letters = fonts(st.weight)
    th = THEME[st.theme]
    rc = RING[(st.ring, st.theme)]
    cx, cy, pts, s, t = geometry(st)
    sx, sy, _ = pts[s]
    tx, ty, _ = pts[t]
    m = ((sx + tx) / 2, (sy + ty) / 2)
    a = math.hypot(tx - sx, ty - sy) / 2
    b = st.aspect * a
    u = ((tx - sx) / (2 * a), (ty - sy) / (2 * a))
    f = (-u[1], u[0])
    # 'lower': the half on the side of the chord facing down the page is in front
    if (f[1] < 0) == (st.front == 'lower'):
        f = (-f[0], -f[1])
    mark_bottom = cy + st.R + st.node_r + 8
    H = mark_bottom + st.word_gap + st.cap + 44
    d, body = [], []

    # --- paints ----------------------------------------------------------
    d.append(f'<linearGradient id="silver-edge" gradientUnits="userSpaceOnUse" '
             f'x1="{f2(cx - st.R)}" y1="{f2(cy - st.R)}" x2="{f2(cx + 0.6 * st.R)}" '
             f'y2="{f2(cy + st.R)}">{stops(th["edge"])}</linearGradient>')
    d.append(f'<linearGradient id="silver-ring" x1="0.3" y1="0" x2="0.7" y2="1">'
             f'{stops(th["ring"])}</linearGradient>')
    d.append(f'<radialGradient id="silver-sphere" cx="0.38" cy="0.32" r="0.72" fx="0.34" '
             f'fy="0.28">{stops(th["sphere"])}</radialGradient>')

    # Two masks, both functions of the distance xi from the chord alone.
    # depth dims the ring from 1 at the front apex to 0.5 at the back apex;
    # step hands the ring from the front layer to the back one inside the
    # terminals' holes, the only places where the ring meets the chord.
    soft = 0.45 * (st.node_r - st.node_w / 2)

    def depth(xi):
        v = max(0.0, min(1.0, (b - xi) / (2 * b)))
        return 1.0 - 0.5 * v ** 1.3

    def step(xi):
        return max(0.0, min(1.0, 0.5 + 0.5 * xi / soft))

    d.append(fgrad('g-front', m, f, b + 60, lambda xi: depth(xi) * step(xi)))
    d.append(fgrad('g-back', m, f, b + 60, lambda xi: depth(xi) * (1 - step(xi))))
    blur = lambda i, sd: (f'<filter id="{i}" filterUnits="userSpaceOnUse" x="-200" y="-200" '
                          f'width="{W + 400}" height="{f2(H + 400)}"><feGaussianBlur stdDeviation="{sd}"/></filter>')
    d.append(blur('blur-wide', 12))
    d.append(blur('blur-mid', 3.5))

    # The crisp back half is cut where the graph lies over it: the edges,
    # the annulus of an open node (not its hole), the disc of a sphere.
    edge_segs = [(pts[k][0], pts[k][1], pts[(k + 1) % st.n][0], pts[(k + 1) % st.n][1])
                 for k in range(st.n)]
    ko = ''.join(f'<line x1="{f2(x1)}" y1="{f2(y1)}" x2="{f2(x2)}" y2="{f2(y2)}"/>'
                 for x1, y1, x2, y2 in edge_segs)
    open_node = lambda k: st.nodes == 'open' or (st.nodes == 'mixed' and k in (s, t))
    ko_nodes = ''.join(
        f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{f2(st.node_r)}" fill="none" '
        f'stroke-width="{f2(st.node_w + 2 * st.gap)}"/>' if open_node(k) else
        f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{f2(st.node_r + st.node_w / 2 + st.gap)}" stroke="none"/>'
        for k, (x, y, _) in enumerate(pts))
    full = f'<rect width="{W}" height="{f2(H)}" fill="url(#g-%s)"/>'
    mask = lambda i, content: (f'<mask id="{i}" maskUnits="userSpaceOnUse" x="0" y="0" '
                               f'width="{W}" height="{f2(H)}">{content}</mask>')
    d.append(mask('m-back', full % 'back' +
                  f'<g stroke="#000" fill="none" stroke-width="{f2(st.edge_w + 2 * st.gap)}">{ko}</g>'
                  f'<g stroke="#000" fill="#000">{ko_nodes}</g>'))
    d.append(mask('m-back-soft', full % 'back'))
    d.append(mask('m-front', full % 'front'))

    # --- ring ------------------------------------------------------------
    k = st.tube
    g = th['glow']
    bnd = lambda wf, wb, ws: band(m, a, b, wf * k, wb * k, ws * k, u, f)
    ev = 'fill-rule="evenodd"'

    def glow(pal):
        return (f'<path {ev} d="{bnd(30, 16, 22)}" fill="{pal["glow"]}" opacity="{0.55 * g:.2f}" filter="url(#blur-wide)"/>'
                f'<path {ev} d="{bnd(13, 6.5, 9.5)}" fill="{pal["mid"]}" opacity="{0.95 * g:.2f}" filter="url(#blur-mid)"/>')

    cw = th['core']

    def crisp(pal):
        return (f'<path {ev} d="{bnd(7.5, 3.6, 5.4)}" fill="{pal["tube"]}"/>'
                f'<path {ev} d="{bnd(3.6 * cw, 1.6 * cw, 2.5 * cw)}" fill="{pal["core"]}"/>')

    back_pal = dict(rc, core=rc['tube'], tube=rc['mid'])
    body.append(f'<g id="hop-back"><g mask="url(#m-back-soft)">{glow(back_pal)}</g>'
                f'<g mask="url(#m-back)">{crisp(back_pal)}</g></g>')

    # The terminals shine: a halo behind them, and a softer light over them.
    ends = ((sx, sy), (tx, ty))
    if st.shine:
        a0 = st.shine * g
        d.append(f'<radialGradient id="shine">'
                 f'<stop offset="0" stop-color="{rc["core"]}" stop-opacity="{min(1, 0.9 * a0):.2f}"/>'
                 f'<stop offset="0.16" stop-color="{rc["mid"]}" stop-opacity="{min(1, 0.75 * a0):.2f}"/>'
                 f'<stop offset="0.45" stop-color="{rc["glow"]}" stop-opacity="{0.25 * a0:.2f}"/>'
                 f'<stop offset="1" stop-color="{rc["glow"]}" stop-opacity="0"/></radialGradient>')
        rr = 3.4 * st.node_r
        body.append('<g id="shine">' + ''.join(
            f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{f2(rr)}" fill="url(#shine)"/>'
            for x, y in ends) + '</g>')

    # --- graph -----------------------------------------------------------
    lines = []
    for kk, (x1, y1, x2, y2) in enumerate(edge_segs):
        L = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        c1 = st.node_r if open_node(kk) else 0
        c2 = st.node_r if open_node((kk + 1) % st.n) else 0
        lines.append(f'<line x1="{f2(x1 + ux * c1)}" y1="{f2(y1 + uy * c1)}" '
                     f'x2="{f2(x2 - ux * c2)}" y2="{f2(y2 - uy * c2)}"/>')
    body.append(f'<g id="edges" stroke="url(#silver-edge)" stroke-width="{f2(st.edge_w)}">'
                f'{"".join(lines)}</g>')
    nd = []
    for kk, (x, y, _) in enumerate(pts):
        if open_node(kk):
            nd.append(f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{f2(st.node_r)}" fill="none" '
                      f'stroke="url(#silver-ring)" stroke-width="{f2(st.node_w)}"/>')
        else:
            nd.append(f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{f2(st.node_r + st.node_w / 2)}" '
                      f'fill="url(#silver-sphere)"/>')
    body.append(f'<g id="nodes">{"".join(nd)}</g>')
    if st.shine:
        d.append(f'<radialGradient id="lit"><stop offset="0.35" stop-color="{rc["core"]}" '
                 f'stop-opacity="{0.55 * st.shine * g:.2f}"/><stop offset="1" stop-color="{rc["mid"]}" '
                 f'stop-opacity="0"/></radialGradient>')
        rl = st.node_r + st.node_w
        body.append('<g id="lit">' + ''.join(
            f'<circle cx="{f2(x)}" cy="{f2(y)}" r="{f2(rl)}" fill="url(#lit)"/>' for x, y in ends)
            + '</g>')
    body.append(f'<g id="hop-front" mask="url(#m-front)">{glow(rc)}{crisp(rc)}</g>')
    if st.glint:
        d.append('<radialGradient id="glint"><stop offset="0" stop-color="#FFFFFF"/>'
                 f'<stop offset="0.3" stop-color="{rc["core"]}" stop-opacity="0.75"/>'
                 f'<stop offset="1" stop-color="{rc["tube"]}" stop-opacity="0"/></radialGradient>')
        gl, gw = st.glint * 2.8 * st.node_r, 0.13 * st.node_r
        rays = ''.join(
            f'<ellipse cx="{f2(x)}" cy="{f2(y)}" rx="{f2(rx)}" ry="{f2(ry)}" fill="url(#glint)"/>'
            for x, y in ends for rx, ry in ((gl, gw), (gw, gl)))
        body.append(f'<g id="glint">{rays}</g>')

    # --- layout ----------------------------------------------------------
    vb = (0, 0, W, H)
    ext = st.R + st.node_r + st.node_w / 2
    word = lambda y0, y1: (f'<linearGradient id="word" gradientUnits="userSpaceOnUse" x1="0" '
                           f'y1="{f2(y0)}" x2="0" y2="{f2(y1)}">{stops(th["word"])}</linearGradient>')
    if st.layout == 'stacked':
        wy = mark_bottom + st.word_gap
        d.append(word(wy, wy + st.cap))
        body.append(f'<path id="wordmark" fill="url(#word)" '
                    f'd="{letters.path("GZL", st.cap, st.tracking, cx, wy)}"/>')
    elif st.layout == 'mark':
        mg = 34
        vb = (cx - ext - mg, cy - ext - mg, 2 * (ext + mg), 2 * (ext + mg))
    elif st.layout == 'icon':
        side = 2 * (ext + 62)
        vb = (cx - side / 2, cy - side / 2, side, side)
        d.append(f'<radialGradient id="tile" gradientUnits="userSpaceOnUse" cx="{f2(cx)}" '
                 f'cy="{f2(cy - 0.35 * side)}" r="{f2(1.05 * side)}">'
                 f'<stop offset="0" stop-color="#1B2B48"/><stop offset="0.55" stop-color="#0C1426"/>'
                 f'<stop offset="1" stop-color="#060A14"/></radialGradient>')
        rx = 0.2237 * side
        body.insert(0, f'<rect x="{f2(vb[0])}" y="{f2(vb[1])}" width="{f2(side)}" height="{f2(side)}" '
                       f'rx="{f2(rx)}" fill="url(#tile)"/>'
                       f'<rect x="{f2(vb[0] + 1)}" y="{f2(vb[1] + 1)}" width="{f2(side - 2)}" '
                       f'height="{f2(side - 2)}" rx="{f2(rx - 1)}" fill="none" stroke="#fff" '
                       f'stroke-opacity="0.08" stroke-width="2"/>')
    elif st.layout == 'horizontal':
        mg = 34
        cap, tracking = 0.6 * ext, 0.2
        wx = cx + ext + 0.34 * ext
        wy = cy - cap / 2
        d.append(word(wy, wy + cap))
        body.append(f'<path id="wordmark" fill="url(#word)" '
                    f'd="{letters.path("GZL", cap, tracking, wx, wy, "left")}"/>')
        right = wx + letters.ink_width('GZL', cap, tracking) + mg
        vb = (cx - ext - mg, cy - ext - mg, right - (cx - ext - mg), 2 * (ext + mg))
    elif st.layout == 'wide':
        mg = 34
        cap, tracking = 0.6 * ext, 0.2
        width = letters.ink_width('GZL', cap, tracking)
        name = fonts(st.name_weight)
        # the name runs flush with GZL on both sides; ink width is linear in cap
        ncap = width / name.ink_width(NAME, 1.0, st.name_tracking)
        gap = st.name_gap * cap
        wx = cx + ext + 0.34 * ext
        wy = cy - (cap + gap + ncap) / 2
        d.append(word(wy, wy + cap))
        body.append(f'<path id="wordmark" fill="url(#word)" '
                    f'd="{letters.path("GZL", cap, tracking, wx, wy, "left")}"/>')
        body.append(f'<path id="name" fill="{th["name"]}" '
                    f'd="{name.path(NAME, ncap, st.name_tracking, wx, wy + cap + gap, "left")}"/>')
        right = wx + width + mg
        vb = (cx - ext - mg, cy - ext - mg, right - (cx - ext - mg), 2 * (ext + mg))
    else:
        raise ValueError(f'unknown layout {st.layout!r}')

    vbs = ' '.join(f2(v) for v in vb)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vbs}" '
            f'width="{f2(vb[2])}" height="{f2(vb[3])}" role="img" aria-label="{TITLE}">'
            f'<title>{TITLE}</title><defs>{"".join(d)}</defs>{"".join(body)}</svg>')


LOGO = Style()
# The icon is drawn heavier, so that it still reads at 32 px.
ICON = replace(LOGO, layout='icon', node_r=24, node_w=12, edge_w=12, tube=1.9, gap=5)

#: The README logo; these two files are in the repository.
FILES = {
    'gzl-logo-wide-dark.svg': replace(LOGO, layout='wide'),
    'gzl-logo-wide-light.svg': replace(LOGO, layout='wide', theme='light'),
}
#: The rest of the set, written with --all.
MORE = {
    'gzl-logo-dark.svg': LOGO,
    'gzl-logo-light.svg': replace(LOGO, theme='light'),
    'gzl-logo-horizontal-dark.svg': replace(LOGO, layout='horizontal'),
    'gzl-logo-horizontal-light.svg': replace(LOGO, layout='horizontal', theme='light'),
    'gzl-mark-dark.svg': replace(LOGO, layout='mark'),
    'gzl-mark-light.svg': replace(LOGO, layout='mark', theme='light'),
    'gzl-icon.svg': ICON,
}


def font_dir() -> Path:
    kpse = shutil.which('kpsewhich') or '/Library/TeX/texbin/kpsewhich'
    try:
        found = subprocess.run([kpse, 'Montserrat-Medium.otf'], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        found = ''
    if not found:
        raise SystemExit('Montserrat not found: install it (TeX Live ships it) '
                         'or pass --font-dir')
    return Path(found).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--font-dir', type=Path, help='directory holding Montserrat-<Weight>.otf')
    ap.add_argument('--out', type=Path, default=HERE, help='output directory (default: docs/logo)')
    ap.add_argument('--all', action='store_true', help='also write the rest of the logo set')
    args = ap.parse_args()
    fonts = Fonts(args.font_dir or font_dir())
    for name, st in (FILES | MORE if args.all else FILES).items():
        path = args.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build(st, fonts) + '\n', encoding='utf-8')
        print(path)

if __name__ == '__main__':
    main()
