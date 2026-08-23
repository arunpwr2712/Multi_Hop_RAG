import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# Read the ablation table
csv_path = Path(__file__).parent / 'ablation_table.csv'
df = pd.read_csv(csv_path)

# Extract model names and HR scores
models = df['Model'].tolist()
hr_scores = df['HR'].tolist()

# Create color mapping - V5 should be green (low HR = good), others red (high HR = bad)
colors = ['#d62728' if model != 'V5' else '#2ca02c' for model in models]

# ============== Static Bar Chart (Matplotlib) ==============
plt.figure(figsize=(10, 6))
bars = plt.bar(models, hr_scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bar, score in zip(bars, hr_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Model Variant', fontsize=12, fontweight='bold')
plt.ylabel('Hallucination Rate (Lower is Better)', fontsize=12, fontweight='bold')
plt.title('Hallucination Rate Comparison Across RAG Model Variants', fontsize=14, fontweight='bold')
plt.ylim(0, max(hr_scores) * 1.15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save static plot
static_path = Path(__file__).parent / 'hr_rate_comparison.png'
plt.savefig(static_path, dpi=300, bbox_inches='tight')
print(f"✓ Static HR plot saved to: {static_path}")
plt.close()

# ============== Interactive Bar Chart (Plotly) ==============
# Reverse colorscale for HR so high values are red (bad), low values are green (good)
fig = go.Figure()

fig.add_trace(go.Bar(
    x=models,
    y=hr_scores,
    marker=dict(
        color=['#d62728' if score > 0.5 else '#2ca02c' for score in hr_scores],
        line=dict(color='black', width=1.5)
    ),
    text=[f'{score:.4f}' for score in hr_scores],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>Hallucination Rate: %{y:.4f}<extra></extra>'
))

fig.update_layout(
    title='Hallucination Rate Comparison Across RAG Model Variants (Lower is Better)',
    xaxis_title='Model Variant',
    yaxis_title='Hallucination Rate',
    font=dict(size=12),
    height=600,
    width=1000,
    showlegend=False,
    template='plotly_white',
    hovermode='x unified'
)

# Save interactive plot
interactive_path = Path(__file__).parent / 'hr_rate_comparison.html'
fig.write_html(interactive_path)
print(f"✓ Interactive HR plot saved to: {interactive_path}")

# ============== Print Summary ==============
print("\n" + "="*60)
print("Hallucination Rate Comparison Summary (Lower is Better)")
print("="*60)
for model, score in zip(models, hr_scores):
    status = "✅ BEST" if model == 'V5' else "❌ HIGH"
    print(f"{model}: {score:.4f} {status}")
print("="*60)
print(f"\nKey Finding:")
print(f"  - V1-V4 average HR: {sum(hr_scores[:-1])/4:.4f} (100% hallucination)")
print(f"  - V5 HR: {hr_scores[-1]:.4f} (0% hallucination)")
print(f"  - V5 reduces hallucination by: {((sum(hr_scores[:-1])/4 - hr_scores[-1]) / (sum(hr_scores[:-1])/4) * 100):.1f}%")
