import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("jacobian_stats.csv")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("deep")

stats = ["real", "imag", "trace", "det", "frob"]
ylabels = {
    "real": "Mean Re(λ)",
    "imag": "Mean |Im(λ)|",
    "trace": "Trace",
    "det": "Determinant",
    "frob": "Frobenius Norm"
}

fig, axs = plt.subplots(len(stats), 1, figsize=(6, 8), sharex=True)

for ax, stat, color in zip(axs, stats, palette):
    mean_col = f"{stat}_mean"
    std_col = f"{stat}_std"

    ax.plot(df["step"], df[mean_col], color=color, label=ylabels[stat], linewidth=1.8)
    ax.fill_between(
        df["step"],
        df[mean_col] - df[std_col],
        df[mean_col] + df[std_col],
        color=color,
        alpha=0.2
    )
    ax.set_ylabel(ylabels[stat])
    ax.tick_params(axis="y", labelsize=10)

axs[-1].set_xlabel("Input Step")

plt.tight_layout(pad=0.5)
plt.savefig("jacobian_stats.pdf", format="pdf")