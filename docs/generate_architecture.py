"""
Generate architecture diagram for GeneticLLM project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'data': '#4ECDC4',      # Teal
    'process': '#45B7D1',   # Blue
    'model': '#96CEB4',     # Green
    'output': '#FFEAA7',    # Yellow
    'hub': '#DDA0DD',       # Plum
    'arrow': '#2C3E50',     # Dark gray
}

def draw_box(ax, x, y, width, height, text, color, fontsize=9):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            wrap=True)

def draw_arrow(ax, start, end, color='#2C3E50'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax.text(7, 9.5, 'GeneticLLM Architecture',
        ha='center', va='center', fontsize=18, fontweight='bold')

# ============ DATA SOURCES (Left) ============
ax.text(1.5, 8.5, 'Data Sources', ha='center', fontsize=12, fontweight='bold', color='#2C3E50')

draw_box(ax, 0.3, 7.2, 2.4, 0.8, 'PubMedQA\n(454)', colors['data'], 8)
draw_box(ax, 0.3, 6.2, 2.4, 0.8, 'MMLU Biology\n(609)', colors['data'], 8)
draw_box(ax, 0.3, 5.2, 2.4, 0.8, 'MedMCQA\n(93,860)', colors['data'], 8)
draw_box(ax, 0.3, 4.2, 2.4, 0.8, 'SciQ\n(3,916)', colors['data'], 8)

# ============ DATA PIPELINE (Center-Left) ============
ax.text(4.5, 8.5, 'Data Pipeline', ha='center', fontsize=12, fontweight='bold', color='#2C3E50')

draw_box(ax, 3.5, 6.5, 2, 1.2, 'download_all.py\n\nFilter &\nAggregate', colors['process'], 8)
draw_box(ax, 3.5, 4.5, 2, 1.2, 'preprocess.py\n\nFormat for\nInstruction Tuning', colors['process'], 8)

# Arrows from data sources to download_all.py (center of left edge)
# Data sources right edge: x=2.7, download_all.py left edge: x=3.5, center y=7.1
draw_arrow(ax, (2.7, 7.6), (3.5, 7.1))   # PubMedQA -> download_all.py
draw_arrow(ax, (2.7, 6.6), (3.5, 7.1))   # MMLU -> download_all.py
draw_arrow(ax, (2.7, 5.6), (3.5, 7.1))   # MedMCQA -> download_all.py
draw_arrow(ax, (2.7, 4.6), (3.5, 7.1))   # SciQ -> download_all.py

# Arrow between pipeline stages
draw_arrow(ax, (4.5, 6.5), (4.5, 5.7))

# ============ HUGGINGFACE HUB (Center) ============
draw_box(ax, 6, 5, 2, 1.5, 'HuggingFace Hub\n\nsachinbkale27/\ngenetics-qa\n(89k samples)', colors['hub'], 8)

# Arrow from preprocess to HF
draw_arrow(ax, (5.5, 5.1), (6, 5.75))

# ============ TRAINING (Center-Right) ============
ax.text(9.5, 8.5, 'Training (Colab)', ha='center', fontsize=12, fontweight='bold', color='#2C3E50')

draw_box(ax, 8.5, 7, 2, 1, 'Qwen2-1.5B\nBase Model', colors['model'], 8)
draw_box(ax, 8.5, 5.5, 2, 1.2, 'QLoRA\nFine-tuning\n(4-bit + LoRA)', colors['process'], 8)

# Arrow from HF to training
draw_arrow(ax, (8, 5.75), (8.5, 6.1))

# Arrow from base model to training
draw_arrow(ax, (9.5, 7), (9.5, 6.7))

# ============ OUTPUT (Right) ============
ax.text(12, 8.5, 'Output', ha='center', fontsize=12, fontweight='bold', color='#2C3E50')

draw_box(ax, 11, 6.5, 2, 1.2, 'GeneticLLM\nLoRA Adapters\n(~50MB)', colors['output'], 8)

# Arrow from training to output
draw_arrow(ax, (10.5, 6.1), (11, 7.1))

# ============ INFERENCE (Bottom Right) ============
draw_box(ax, 11, 4.5, 2, 1.2, 'Inference\n\nquery.py\nInteractive Q&A', colors['process'], 8)

# Arrow from output to inference
draw_arrow(ax, (12, 6.5), (12, 5.7))

# ============ EVALUATION (Bottom Center) ============
draw_box(ax, 8.5, 3, 2, 1.2, 'Evaluation\n\nBLEU, ROUGE\nTerminology F1', colors['process'], 8)

# Arrow from inference to evaluation
draw_arrow(ax, (11, 5.1), (10.5, 4))

# ============ LEGEND ============
legend_y = 1.5
ax.text(1, legend_y + 0.8, 'Legend:', fontsize=10, fontweight='bold')

legend_items = [
    ('Data Sources', colors['data']),
    ('Processing', colors['process']),
    ('Model', colors['model']),
    ('HuggingFace Hub', colors['hub']),
    ('Output', colors['output']),
]

for i, (label, color) in enumerate(legend_items):
    x = 1 + (i * 2.5)
    box = FancyBboxPatch(
        (x, legend_y), 0.4, 0.4,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.5, legend_y + 0.2, label, fontsize=8, va='center')

# ============ STATS BOX ============
stats_text = """Training: 88,955 samples
Validation: 9,884 samples
Base Model: Qwen2-1.5B-Instruct
Method: QLoRA (4-bit quantization)
Hardware: Free Colab T4 GPU"""

ax.text(1, 0.3, stats_text, fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2C3E50', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/architecture.jpg', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('docs/architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: docs/architecture.jpg and docs/architecture.png")
