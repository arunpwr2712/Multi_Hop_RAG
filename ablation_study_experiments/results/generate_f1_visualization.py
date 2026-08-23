import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# Read the ablation table
csv_path = Path(__file__).parent / 'ablation_table.csv'
df = pd.read_csv(csv_path)

# Extract model names and F1 scores
models = df['Model'].tolist()
f1_scores = df['F1'].tolist()

# Create color mapping - highlight V5 as best model
colors = ['#1f77b4' if model != 'V5' else '#2ca02c' for model in models]

# ============== Static Bar Chart (Matplotlib) ==============
plt.figure(figsize=(10, 6))
bars = plt.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Model Variant', fontsize=12, fontweight='bold')
plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
plt.title('F1 Score Comparison Across RAG Model Variants', fontsize=14, fontweight='bold')
plt.ylim(0, max(f1_scores) * 1.15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save static plot
static_path = Path(__file__).parent / 'f1_score_comparison.png'
plt.savefig(static_path, dpi=300, bbox_inches='tight')
print(f"✓ Static plot saved to: {static_path}")
plt.close()

# ============== Interactive Bar Chart (Plotly) ==============
fig = go.Figure()

fig.add_trace(go.Bar(
    x=models,
    y=f1_scores,
    marker=dict(
        color=f1_scores,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="F1 Score"),
        line=dict(color='black', width=1.5)
    ),
    text=[f'{score:.4f}' for score in f1_scores],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>F1 Score: %{y:.4f}<extra></extra>'
))

fig.update_layout(
    title='F1 Score Comparison Across RAG Model Variants',
    xaxis_title='Model Variant',
    yaxis_title='F1 Score',
    font=dict(size=12),
    height=600,
    width=1000,
    showlegend=False,
    template='plotly_white',
    hovermode='x unified'
)

# Save interactive plot
interactive_path = Path(__file__).parent / 'f1_score_comparison.html'
fig.write_html(interactive_path)
print(f"✓ Interactive plot saved to: {interactive_path}")

# ============== Print Summary ==============
print("\n" + "="*50)
print("F1 Score Comparison Summary")
print("="*50)
for model, score in zip(models, f1_scores):
    indicator = " ← Best Model" if model == 'V5' else ""
    print(f"{model}: {score:.4f}{indicator}")
print("="*50)
print(f"\nV5 outperforms baselines by:")
baseline_avg = sum(f1_scores[:-1]) / 4
print(f"  - {((f1_scores[-1] - baseline_avg) / baseline_avg * 100):.1f}% vs average of V1-V4")
print(f"  - {((f1_scores[-1] - f1_scores[1]) / f1_scores[1] * 100):.1f}% vs best baseline (V2)")
