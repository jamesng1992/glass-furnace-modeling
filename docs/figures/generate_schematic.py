"""Generate a glass furnace schematic diagram for the README."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')

# ── Colors ──────────────────────────────────────────────────────────
c_furnace   = '#D4A574'   # warm tan for furnace walls
c_glass     = '#E8843C'   # orange for molten glass
c_batch     = '#C9B896'   # beige for batch blanket
c_flame     = '#FF6347'   # flame
c_arrow_in  = '#2E86C1'   # blue for inputs
c_arrow_out = '#E74C3C'   # red for outputs
c_neural    = '#8E44AD'   # purple for neural ODE
c_bg        = '#FAFAFA'

fig.patch.set_facecolor(c_bg)

# ── Main furnace body ──────────────────────────────────────────────
furnace = FancyBboxPatch((2, 0.8), 8, 4.2, boxstyle="round,pad=0.15",
                          facecolor=c_furnace, edgecolor='#5D4037',
                          linewidth=2.5)
ax.add_patch(furnace)

# Crown (roof)
crown_x = [1.85, 6, 10.15]
crown_y = [5.0, 5.8, 5.0]
ax.fill(crown_x, crown_y, color='#B0856A', edgecolor='#5D4037', linewidth=2.5)
ax.text(6, 5.35, 'Crown', fontsize=9, ha='center', va='center',
        fontweight='bold', color='#3E2723')

# Molten glass pool
glass = FancyBboxPatch((2.15, 0.95), 7.7, 2.2, boxstyle="round,pad=0.08",
                         facecolor=c_glass, edgecolor='#BF360C',
                         linewidth=1.5, alpha=0.85)
ax.add_patch(glass)
ax.text(6, 2.0, 'Molten Glass Pool', fontsize=11, ha='center', va='center',
        fontweight='bold', color='white')

# Glass level indicator line
ax.plot([2.15, 9.85], [3.15, 3.15], '--', color='white', linewidth=1.5, alpha=0.7)
ax.text(6, 3.35, '$h$ (glass level)', fontsize=9, ha='center', va='bottom',
        color='white', fontstyle='italic')

# Batch blanket on top of glass
batch_x = [2.3, 2.8, 4.5, 5.0, 4.8]
batch_y = [3.15, 3.6, 3.55, 3.3, 3.15]
ax.fill(batch_x, batch_y, color=c_batch, edgecolor='#8D6E63', linewidth=1.5)
ax.text(3.5, 3.55, 'Batch\nBlanket', fontsize=7.5, ha='center', va='center',
        color='#5D4037', fontweight='bold')

# Combustion / flame zone
for i, cx in enumerate([5.5, 7.0, 8.5]):
    flame = patches.Polygon(
        [[cx-0.2, 4.0], [cx, 4.7], [cx+0.2, 4.0]],
        closed=True, facecolor=c_flame, edgecolor='#FF8F00',
        linewidth=1, alpha=0.7 + 0.1*i
    )
    ax.add_patch(flame)
ax.text(7, 4.55, 'Combustion', fontsize=8, ha='center', va='center',
        color='#BF360C', fontweight='bold')

# ── Batch feed (input u₁) ──────────────────────────────────────────
ax.annotate('', xy=(2.3, 3.4), xytext=(0.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=c_arrow_in, lw=2.5))
ax.text(0.3, 5.7, '$u_1$: Batch Feed\n(t/h)', fontsize=10, ha='left',
        va='bottom', color=c_arrow_in, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=c_arrow_in, alpha=0.9))

# ── Pull rate (output u₂) ──────────────────────────────────────────
ax.annotate('', xy=(12.0, 2.0), xytext=(9.85, 2.0),
            arrowprops=dict(arrowstyle='->', color=c_arrow_out, lw=2.5))
ax.text(12.2, 2.0, '$u_2$: Pull Rate\n(m³/h)', fontsize=10, ha='left',
        va='center', color=c_arrow_out, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=c_arrow_out, alpha=0.9))

# Throat / waist
throat = FancyBboxPatch((9.85, 1.3), 0.8, 1.4, boxstyle="round,pad=0.05",
                          facecolor='#E8843C', edgecolor='#5D4037',
                          linewidth=2, alpha=0.8)
ax.add_patch(throat)
ax.text(10.25, 2.5, 'Throat', fontsize=7, ha='center', va='bottom',
        color='#5D4037', fontweight='bold', rotation=0)

# Refiner section
refiner = FancyBboxPatch((10.65, 1.1), 1.6, 1.8, boxstyle="round,pad=0.1",
                           facecolor=c_furnace, edgecolor='#5D4037',
                           linewidth=2)
ax.add_patch(refiner)
refiner_glass = FancyBboxPatch((10.75, 1.2), 1.4, 1.0, boxstyle="round,pad=0.05",
                                 facecolor='#E8843C', edgecolor='#BF360C',
                                 linewidth=1, alpha=0.75)
ax.add_patch(refiner_glass)
ax.text(11.45, 2.2, 'Refiner', fontsize=8, ha='center', va='center',
        color='#5D4037', fontweight='bold')

# ── Transport delay chain annotation ──────────────────────────────
delay_y = 0.35
for i, (x, label) in enumerate([(3.0, '$z_1$'), (4.5, '$z_2$'),
                                  (6.0, '$z_3$'), (7.5, '$z_4$')]):
    box = FancyBboxPatch((x-0.35, delay_y-0.2), 0.7, 0.4,
                           boxstyle="round,pad=0.05",
                           facecolor='#FFF9C4', edgecolor='#F9A825',
                           linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, delay_y, label, fontsize=9, ha='center', va='center',
            fontweight='bold', color='#E65100')
    if i < 3:
        ax.annotate('', xy=(x+0.85, delay_y), xytext=(x+0.35, delay_y),
                    arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.5))

ax.text(5.25, -0.1, 'Transport Delay Chain (Erlang-4)', fontsize=9,
        ha='center', va='top', color='#E65100', fontstyle='italic')

# Arrow from z4 to qm
ax.annotate('', xy=(9.0, delay_y), xytext=(8.15, delay_y),
            arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.5))
qm_box = FancyBboxPatch((8.9, delay_y-0.2), 0.85, 0.4,
                           boxstyle="round,pad=0.05",
                           facecolor='#FFCCBC', edgecolor='#E64A19',
                           linewidth=1.5)
ax.add_patch(qm_box)
ax.text(9.32, delay_y, '$q_m$', fontsize=9, ha='center', va='center',
        fontweight='bold', color='#BF360C')

# ── Neural ODE correction block ───────────────────────────────────
nn_box = FancyBboxPatch((11.0, 4.2), 2.5, 1.6, boxstyle="round,pad=0.15",
                          facecolor='#F3E5F5', edgecolor=c_neural,
                          linewidth=2, linestyle='--')
ax.add_patch(nn_box)
ax.text(12.25, 5.25, 'Neural ODE', fontsize=10, ha='center', va='center',
        fontweight='bold', color=c_neural)
ax.text(12.25, 4.8, r'$f_\theta(\mathbf{x}, \mathbf{u})$', fontsize=11,
        ha='center', va='center', color=c_neural)
ax.text(12.25, 4.4, 'Learned Correction', fontsize=7.5, ha='center',
        va='center', color=c_neural, fontstyle='italic')

ax.annotate('', xy=(10.15, 4.9), xytext=(11.0, 4.9),
            arrowprops=dict(arrowstyle='->', color=c_neural, lw=2,
                            linestyle='--'))

# ── Title ──────────────────────────────────────────────────────────
ax.text(7, 6.7, 'Glass Melting Furnace — Hybrid Neural ODE Model',
        fontsize=14, ha='center', va='center', fontweight='bold',
        color='#263238')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'glass_furnace_schematic.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=c_bg)
print(f"Saved schematic to {out_path}")
plt.close()
