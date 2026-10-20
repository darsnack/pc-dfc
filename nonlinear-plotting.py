"""Plot the nonlinear experiment from its saved simulation_data.pkl."""
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import OmegaConf


def fit_pca(arrays):
    """Fit PCA on concatenated arrays; return (project, mean, components)."""
    combined = np.concatenate(arrays, axis=0)
    mean = combined.mean(axis=0)
    n_dim = combined.shape[1]
    if n_dim == 1:
        components = np.array([[1.0], [0.0]])
        def project(t):
            return np.column_stack([np.asarray(t) - mean, np.zeros(len(t))])
    else:
        _, _, vt = np.linalg.svd(combined - mean, full_matrices=False)
        components = vt[:2]
        def project(t):
            return (np.asarray(t) - mean) @ components.T
    return project, mean, components

def rs_label(rs, break_line=False):
    if rs is None:
        return "relinearize\nnever" if break_line else "relinearize never"
    return f"relinearize every\n{rs} steps" if break_line else f"relinearize every {rs} steps"

def relinearization_indices(rs, total_timesteps):
    """Indices into the history array where re-linearization occurred."""
    step_size = rs or total_timesteps
    return list(range(0, total_timesteps, step_size))


@hydra.main(config_path="./configs", config_name="nonlinear-plotting", version_base=None)
def main(cfg) -> None:
    experiment_dir = Path(hydra.utils.to_absolute_path(cfg.experiment_dir))
    with (experiment_dir / "simulation_data.pkl").open("rb") as f:
        data = pickle.load(f)
    experiment_cfg = OmegaConf.create(data["config"])
    pcn_hist = data["pcn_hist"]
    approx_hists = data["approx_hists"]
    approx_relin_counts = data["approx_relin_counts"]
    approx_segment_stats = data["approx_segment_stats"]
    delta_flow = data["delta_flow"]
    rs_sweep_vals = list(approx_hists)
    traj_plot_vals = list(cfg.trajectory_plot_vals)
    layer_dims = [experiment_cfg.model.input_dim,
                  *experiment_cfg.model.hidden_dims,
                  experiment_cfg.model.output_dim]
    n_layers = len(layer_dims) - 1
    layer_labels = (
        [f"Hidden 1 (n={layer_dims[1]})"]
        + [f"Hidden {i+2} (n={layer_dims[i+2]})" for i in range(len(layer_dims) - 3)]
        + [f"Output (n={layer_dims[-1]})"]
    )
    rs_labels = ["never" if rs is None else str(rs) for rs in rs_sweep_vals]
    out_dir = experiment_dir
    sns.set_context("paper", font_scale=2)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # Final loss plot
    layer_palette = dict(zip(layer_labels, sns.light_palette("seagreen", n_layers + 1)[1:]))
    fig_loss, ax_loss = plt.subplots(figsize=(max(6, 0.7 * len(rs_sweep_vals) + 3.5), 5))
    for layer_idx in range(n_layers):
        pcn_final = pcn_hist[layer_idx][-1]
        final_diffs = [
            float(np.linalg.norm(approx_hists[rs][layer_idx][-1] - pcn_final))
            for rs in rs_sweep_vals
        ]
        sns.lineplot(x=rs_labels, y=final_diffs,
                     marker="o",
                     linewidth=1.5,
                     sort=False,
                     color=layer_palette[layer_labels[layer_idx]], label=layer_labels[layer_idx],
                     ax=ax_loss)
    ax_loss.set_xlabel("Relinearization Frequency (steps)")
    ax_loss.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx}^T - \mathbf{v}_\mathrm{PCN}^T\|$")
    ax_loss.set_yscale("log")
    ax_loss.set_title(f"Final activation diff  |  noise={experiment_cfg.noise:.2f}", pad=20, fontweight="bold")
    ax_loss.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.3, 1))
    plt.xticks(rotation=45, ha="right")
    sns.despine(ax=ax_loss)
    # Num. of relinearizations plot
    rs_relin_only = [rs for rs in rs_sweep_vals if rs is not None]
    rs_labels_relin_only = [str(rs) for rs in rs_relin_only]
    relin_counts = [approx_relin_counts[rs] for rs in rs_relin_only]
    ax_relin = ax_loss.twinx()
    ax_relin.plot(rs_labels_relin_only, relin_counts,
                  marker="s", linestyle="--", color="#7a7a7a", linewidth=1.2, markersize=5, zorder=5)
    ax_relin.set_ylim(0.9, None)
    ax_relin.set_yscale("log")
    ax_relin.set_ylabel("# of Relinearizations")
    relin_handle = plt.Line2D([], [],
                              color="#7a7a7a",
                              marker="s", markersize=5,
                              linestyle="--", linewidth=1.2,
                              label="Re-linearizations")
    ax_relin.legend(handles=[relin_handle], frameon=False, loc="lower left", bbox_to_anchor=(1.3, 0))
    plt.xticks(rotation=45, ha="right")
    fig_loss.tight_layout()
    out_loss = out_dir / "final_loss.png"
    fig_loss.savefig(out_loss, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved {out_loss}")
    plt.close(fig_loss)


    # Early convergence plot
    x_labels = []
    convergence_rate = []
    for label, rs in zip(rs_labels, rs_sweep_vals):
        if rs is None:
            continue
        segment_stats = approx_segment_stats[rs]
        x_labels.append(label)
        convergence_rate.append(100.0 * np.mean([stat["converged"] for stat in segment_stats]))
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(x_labels) + 2), 3.5))
    sns.barplot(x=x_labels, y=convergence_rate, color="#4daf4a", ax=ax)
    ax.set_xlabel("Relinearization Frequency (steps)")
    ax.set_ylabel("% segments converged")
    y_min = min(convergence_rate)
    y_max = max(convergence_rate)
    pad = max(2.0, 0.12 * max(y_max - y_min, 1.0))
    ax.set_ylim(max(0, y_min - pad), min(100, y_max + pad))
    ax.set_title("Segment convergence rate", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    sns.despine(ax=ax)
    fig.tight_layout()
    out = out_dir / "relinearization_convergence_rate.png"
    fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


    # Per-layer trajectory/deltaflow plots (trajectory_plot_vals subset only)
    for layer_idx in range(n_layers):
        pcn_layer = pcn_hist[layer_idx]          # (T_pcn, n_neurons)
        approx_layers = {rs: approx_hists[rs][layer_idx] for rs in traj_plot_vals}
        title = f"{layer_labels[layer_idx]}  |  noise={experiment_cfg.noise:.2f}"

        # fit pca
        project, _, _ = fit_pca([pcn_layer] + list(approx_layers.values()))
        pcn_proj = project(pcn_layer)

        traj_palette = sns.color_palette()
        pc_color = traj_palette[0]
        approx_color = traj_palette[1]

        # Δflow over time
        fig, ax = plt.subplots(figsize=(8, 3))
        layer_delta = delta_flow[:, layer_idx]
        sns.lineplot(x=np.arange(len(layer_delta)), y=layer_delta,
                     color=layer_palette[layer_labels[layer_idx]], linewidth=1.5, ax=ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel(r"$\|\Delta\mathrm{flow}(t)\|$")
        ax.set_title(title, fontweight="bold")
        sns.despine(ax=ax)
        fig.tight_layout()
        out = out_dir / "deltaflow" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)

        # PCA trajectory — side by side
        n_rs = len(traj_plot_vals)
        fig, axes = plt.subplots(1, n_rs, figsize=(3 * n_rs, 5),
                                  sharex=True, sharey=True)
        if n_rs == 1:
            axes = [axes]
        fig.suptitle(title, fontweight="bold")
        for ax, (rs, approx_layer) in zip(axes, approx_layers.items()):
            proj = project(approx_layer)
            sns.lineplot(x=pcn_proj[:, 0], y=pcn_proj[:, 1],
                         color=pc_color, linewidth=2, zorder=10, sort=False, ax=ax)
            sns.scatterplot(x=[pcn_proj[0, 0]], y=[pcn_proj[0, 1]],
                            marker="o", s=50, facecolors="none", edgecolors=pc_color, linewidth=1.5, zorder=11, ax=ax)
            sns.scatterplot(x=[pcn_proj[-1, 0]], y=[pcn_proj[-1, 1]],
                            marker="+", s=100, color=pc_color, linewidth=1.5, zorder=11, ax=ax)
            sns.lineplot(x=proj[:, 0], y=proj[:, 1],
                         color=approx_color, linewidth=1.5, zorder=9, sort=False, ax=ax)
            sns.scatterplot(x=[proj[0, 0]], y=[proj[0, 1]],
                            marker="o", s=30, facecolors="none", edgecolors=approx_color, linewidth=1.2, ax=ax)
            sns.scatterplot(x=[proj[-1, 0]], y=[proj[-1, 1]],
                            marker="+", s=100, color=approx_color, linewidth=1.5, zorder=12, ax=ax)
            lin_idx = relinearization_indices(rs, int(experiment_cfg.dynamics.clamp_timesteps))
            valid = [i for i in lin_idx if i < len(proj)]
            if valid:
                sns.scatterplot(x=proj[valid, 0], y=proj[valid, 1],
                                marker=".", s=30, color="red", zorder=13, ax=ax)
            ax.set_title(rs_label(rs, break_line=True))
            ax.set_xlabel("PC 1")
            ax.set_box_aspect(1)
            sns.despine(ax=ax, offset=10)
        axes[0].set_ylabel("PC 2")
        pc_handle = plt.Line2D([], [], color=pc_color, linewidth=2, label="PC")
        approx_handle = plt.Line2D([], [], color=approx_color, linewidth=1.5, label="Approx PC")
        relin_handle = plt.Line2D([], [], color="red", marker=".", linestyle="None", markersize=6, label="Relinearization")
        fig.legend(handles=[pc_handle, approx_handle, relin_handle],
                   frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
        out = out_dir / "traj_sidebyside" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
