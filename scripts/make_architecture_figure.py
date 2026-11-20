"""Architecture of the deployed system (Fig. in Section VI). Output: paper/figures/architecture.pdf/.png"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"
FILL = {"user": "#eef4ff", "gcp": "#eefaf0", "ext": "#fff6e5", "off": "#f4f4f5"}


def box(x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=FILL[kind], ec=LINE, lw=0.8))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top", fontsize=6.2, color=MUTED, linespacing=1.25)


def arrow(p, q, label="", dy=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=5.8, color=MUTED)


def poly(pts, label="", at=None):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=INK, lw=0.8)
    arrow(pts[-2], pts[-1])
    if label:
        ax.text(at[0], at[1], label, ha="center", va="bottom", fontsize=5.8, color=MUTED)


# Google Cloud frame
ax.add_patch(FancyBboxPatch((22, 5), 54, 38, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec="#2da44e",
                            lw=0.8, ls=(0, (3, 2))))
ax.text(23.2, 41.8, "Google Cloud", fontsize=6.2, color="#1a7f37", va="top")

box(1, 26, 15, 13, "Store user", "Microsoft Teams", "user")
box(25, 24, 20, 15, "First-contact agent", "Gemini 2.5 Flash:\nintake, self-help\nsteps, confirmation", "gcp")
box(56, 29, 18, 11, "itsm-classifier", "MLP, threshold 0.8\n+ TF-IDF retrieval", "gcp")
box(56, 12, 18, 11, "Knowledge base", "12,848 resolved\nincidents (anon.)", "gcp")
box(25, 11.5, 20, 9, "Structured logs", "per-case evaluation", "gcp")
box(82, 26, 17, 13, "Jira Service Mgmt", "ticket: category, p,\nsummary, cases", "ext")
box(82, 4, 17, 13, "Technician", "starts from a\ncategorized ticket", "ext")
box(1, 4, 15, 13, "Offline pipeline", "taxonomy and\ntraining (III\u2013V)", "off")

arrow((16.6, 32.5), (24.4, 32.5))
ax.text(19.2, 33.4, "chat", ha="center", va="bottom", fontsize=5.8, color=MUTED)
arrow((45.6, 34.5), (55.4, 34.5), "")
ax.text(50.5, 35.3, "classify +\nretrieve", ha="center", va="bottom", fontsize=5.6, color=MUTED, linespacing=1.1)
arrow((65, 28.4), (65, 23.6))
arrow((35, 23.4), (35, 20.6))
arrow((90.5, 25.4), (90.5, 17.6))
poly([(35, 39.6), (35, 47), (90.5, 47), (90.5, 39.6)], "not solved, request or notice \u2192 ticket", (62, 47.4))
poly([(16.6, 8.5), (65, 8.5), (65, 11.4)], "model + KB assets (private bucket)", (40, 8.9))
ax.set_ylim(0, 50)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"architecture.{ext}", bbox_inches="tight", dpi=300)
print(OUT / "architecture.pdf")
