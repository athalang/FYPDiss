import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("samples_agg.csv")
ts = np.linspace(0, 1, len(df))

plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")

plt.plot(ts, df["mean_distance"], label="Mean distance", color="C0")
plt.fill_between(ts,
                 df["mean_distance"] - df["std_distance"],
                 df["mean_distance"] + df["std_distance"],
                 alpha=0.2, color="C0")

plt.xlabel("Interpolation Time")
plt.ylabel("Mean Euclidean Distance from SLERP")
plt.title("Deviation from SLERP During Trajectory")
plt.legend()
plt.tight_layout()

plt.savefig("deviation_plot.pdf")

df = pd.read_csv("samples_raw.csv")

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.3)

ax_main = fig.add_subplot(gs[0, 0])
g = sns.jointplot(
    data=df,
    x="norm",
    y="euclidean_distance",
    kind="hex",
    cmap="viridis_r",
    bins="log",
    mincnt=1,
    height=6,
    edgecolors='none'
)
g.ax_joint.set_xlabel("Quaternion Norm")
g.ax_joint.set_ylabel("Euclidean Distance from SLERP")
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
cbar_ax = g.figure.add_axes([.85, .25, .05, .4])
cb = plt.colorbar(cax=cbar_ax)
cb.set_label("Sample Count (log scale)")

plot_path = "deviation_v_norm.pdf"
plt.savefig(plot_path, bbox_inches="tight")
