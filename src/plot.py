import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# =================================================================
# 0. STYLE SETUP
# =================================================================
# Use a professional and clean plot style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'sans-serif' # Use a clean sans-serif font

# =================================================================
# 1. DATA SETUP (Avg. column removed)
# =================================================================
datasets = ['NQ', 'HotpotQA', 'TriviaQA', 'PopQA', '2wiki', 'Musique', 'Bamboogle']
model_sizes = ['0.5B', '1.5B', '3B']

em_scores = {
    '0.5B': [0.194, 0.114, 0.233, 0.262, 0.160, 0.029, 0.056],
    '1.5B': [0.358, 0.257, 0.510, 0.389, 0.188, 0.057, 0.128],
    '3B':   [0.434, 0.373, 0.584, 0.434, 0.381, 0.137, 0.328]
}

f1_scores = {
    '0.5B': [0.248, 0.158, 0.291, 0.299, 0.199, 0.060, 0.122],
    '1.5B': [0.441, 0.348, 0.589, 0.434, 0.238, 0.124, 0.211],
    '3B':   [0.516, 0.482, 0.665, 0.483, 0.442, 0.207, 0.434]
}

# =================================================================
# 2. PLOTTING LOGIC (With new Morandi color palette)
# =================================================================
fig, ax1 = plt.subplots(figsize=(16, 8))
ax2 = ax1.twinx()

n_datasets = len(datasets)
n_models = len(model_sizes)
bar_width = 0.25
index = np.arange(n_datasets)


# # Bars: Inspired by water lilies and sky (Lilac, Sky Blue, Deep Blue)
morandi_bar_colors = ['#b3aed4', '#75a8d3', '#3d6c9e'] 
# Lines: Inspired by haystacks at sunset (Soft Yellow, Apricot, Burnt Orange)
morandi_line_colors = ['#f5d98f', '#f2ac74', '#d97b53']

# --- Monet Color Palette: Pastel Garden ---
# # Bars: Inspired by willows and greenery (Sage, Teal, Forest Green)
# morandi_bar_colors = ['#a2bca3', '#5e9a9d', '#2e6b5f'] 
# # Lines: Inspired by garden flowers (Pale Rose, Coral, Dusty Pink)
# morandi_line_colors = ['#e4b7ba', '#d48c82', '#b86b77']

# --- Plotting the Bars (EM Scores) ---
for i, model_size in enumerate(model_sizes):
    bar_position = index - bar_width + (i * bar_width)
    ax1.bar(
        bar_position, 
        em_scores[model_size], 
        bar_width, 
        label=f'EM ({model_size})',
        color=morandi_bar_colors[i],
        edgecolor='white' # Add a subtle white edge for better separation
    )

# --- Plotting the Lines (F1 Scores) ---
for i, model_size in enumerate(model_sizes):
    ax2.plot(
        index, 
        f1_scores[model_size], 
        marker='o',
        markersize=8,
        linestyle='--',
        linewidth=2,
        label=f'F1 ({model_size})',
        color=morandi_line_colors[i]
    )

# =================================================================
# 3. STYLING AND LABELS (Enhanced for aesthetics)
# =================================================================
# ax1.set_title('evolver/ Performance Scaling Across Datasets', fontsize=20, pad=25, weight='bold')
ax1.set_xlabel('Datasets', fontsize=24, labelpad=20)
ax1.set_ylabel('EM Score', fontsize=24, color=morandi_bar_colors[2])
ax2.set_ylabel('F1 Score', fontsize=24, color=morandi_line_colors[2], rotation=-90, labelpad=25) # Rotate for better fit

ax1.set_xticks(index)
ax1.set_xticklabels(datasets, fontsize=20)
ax1.tick_params(axis='y', labelsize=20, colors=morandi_bar_colors[2])
ax2.tick_params(axis='y', labelsize=20, colors=morandi_line_colors[2])
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax2.grid(False) # Turn off the grid for the secondary axis to avoid clutter

ax1.set_ylim(0, 0.8) # Set a fixed limit for better comparison
ax2.set_ylim(0, 0.8) # Keep y-axes scales consistent for intuitive comparison

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=20, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

fig.tight_layout()
plt.savefig("./plt_result/evolver/_scaling_results-1.png", dpi=500, bbox_inches='tight')
plt.show()