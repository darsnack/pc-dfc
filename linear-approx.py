"""
Compare PCN vs approx-PC inference dynamics (no weight learning).

Procedure:
  1. Build a small MLP with random weights.
  2. Run PCN free-phase dynamics until convergence (fixed point).
  3. Sweep over noise magnitudes; for each, clamp the output to
     fp_output + noise_direction * magnitude and run both networks.
  4. Plots
       dynamics_layer{l}.png  — per-neuron time series, rows = noise levels
       dynamics_pca.png       — PC1/PC2 trajectories,  rows = noise levels
       dynamics_heatmap.png   — layer × noise heatmap of PCN vs approx-PC error
"""

import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from flax import nnx

from projectlib.models import MLP
from projectlib.dynamics import pc_step, approx_pc_step, current_pc_vjp, run
from projectlib.utils import noop

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compare PCN vs approx-PC dynamics.")
parser.add_argument(
    "--hidden", type=int, nargs="+", default=[8, 6, 4],
    metavar="N",
    help="Hidden layer widths (default: 8 6 4)",
)
parser.add_argument("--seed",        type=int,   default=10)
parser.add_argument("--tau",         type=float, default=10.0)
parser.add_argument("--k",           type=float, default=1.0)
parser.add_argument("--clamp-steps", type=int,   default=1000)
parser.add_argument(
    "--activation", default="identity",
    choices=["identity", "tanh", "relu"],
    help="Hidden-layer activation function (default: identity)",
)
parser.add_argument(
    "--noise-sweep", type=float, nargs="+",
    default=[0.1, 0.5, 1.0, 10.0, 50.0],
    metavar="N",
    help="Noise magnitudes to sweep (default: 0.1 0.3 0.5 0.8 1.2)",
)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────
SEED         = args.seed
LAYER_DIMS   = [1, *args.hidden, 1]
TAU          = args.tau
K            = args.k
NOISE_MAGS   = args.noise_sweep
FREE_MIN     = 10
FREE_MAX     = 2000
CLAMP_STEPS  = args.clamp_steps
TOLERANCE    = 1e-9
N_NOISE      = len(NOISE_MAGS)
N_LAYERS     = len(LAYER_DIMS) - 1   # number of post-input layers

ACTIVATION_FNS = {"identity": noop, "tanh": nnx.tanh, "relu": nnx.relu}
activation_fn  = ACTIVATION_FNS[args.activation]

layer_labels = (
    [f"Hidden {1}  (n={LAYER_DIMS[1]})"]
    + [f"Hidden {i+2}  (n={LAYER_DIMS[i+2]})" for i in range(len(LAYER_DIMS) - 3)]
    + [f"Output  (n={LAYER_DIMS[-1]})"]
)

# ── Build model & random input ────────────────────────────────────────────
rngs  = nnx.Rngs(SEED, params=SEED + 1, activations=SEED + 2)
model = MLP(LAYER_DIMS, activation_fn=activation_fn, rngs=rngs)

rng = jrng.PRNGKey(SEED)
rng, rng_x, rng_n = jrng.split(rng, 3)
x = jrng.normal(rng_x, (1,))

# ── Phase 1 – free phase → fixed point ───────────────────────────────────
_, free_hist, T_free = run(
    model, pc_step, x,
    min_timesteps=FREE_MIN, max_timesteps=FREE_MAX,
    tolerance=TOLERANCE, tau=TAU,
    learn=False, return_act_history=True,
)
print(f"Free-phase converged in {T_free} steps.")

fp_acts   = model.get_activations()
graph_def, fp_state = nnx.split(model)

# Fixed noise direction; magnitude is varied in the sweep
noise_dir = jrng.normal(rng_n, fp_acts[-1].shape)
noise_dir = noise_dir / jnp.linalg.norm(noise_dir)   # unit vector

# VJPs are the same for every noise level (fixed point & weights don't change)
_tmp = nnx.merge(graph_def, fp_state)
vjps = current_pc_vjp(_tmp, x)

def approx_step(model, x, **kwargs):
    return approx_pc_step(model, x, vjps, k=K, **kwargs)

def concat(free, clamped):
    """Stitch free-phase and clamped-phase histories (drop duplicate junction)."""
    return jnp.concatenate([free, clamped[1:]], axis=0)

# ── Phase 2 – sweep over noise magnitudes ────────────────────────────────
sweep_pcn_clamp    = []   # [N_NOISE][N_LAYERS]  arrays [T+1, n]
sweep_approx_clamp = []
sweep_pcn_full     = []   # free + clamped stitched
sweep_approx_full  = []
diff_matrix        = np.zeros((N_NOISE, N_LAYERS))   # for heatmap

for ni, mag in enumerate(NOISE_MAGS):
    rclamp = fp_acts[-1] + noise_dir * mag

    pcn_model    = nnx.merge(graph_def, fp_state)
    approx_model = nnx.merge(graph_def, fp_state)

    _, pcn_ch, _ = run(
        pcn_model, pc_step, x,
        min_timesteps=CLAMP_STEPS, max_timesteps=CLAMP_STEPS,
        tolerance=TOLERANCE, tau=TAU,
        learn=False, rclamp=rclamp, return_act_history=True,
    )
    _, approx_ch, _ = run(
        approx_model, approx_step, x,
        min_timesteps=CLAMP_STEPS, max_timesteps=CLAMP_STEPS,
        tolerance=TOLERANCE, tau=TAU,
        learn=False, rclamp=rclamp, return_act_history=True,
    )

    sweep_pcn_clamp.append([np.array(h) for h in pcn_ch])
    sweep_approx_clamp.append([np.array(h) for h in approx_ch])
    sweep_pcn_full.append([concat(f, c) for f, c in zip(free_hist, pcn_ch)])
    sweep_approx_full.append([concat(f, c) for f, c in zip(free_hist, approx_ch)])

    for l in range(N_LAYERS):
        diff_matrix[ni, l] = np.linalg.norm(
            sweep_pcn_clamp[-1][l][-1] - sweep_approx_clamp[-1][l][-1]
        )

    print(f"  noise={mag:.2f}  |  layer diffs: "
          + "  ".join(f"L{l+1}={diff_matrix[ni,l]:.4f}" for l in range(N_LAYERS)))

t_axis  = np.arange(sweep_pcn_full[0][0].shape[0])
t_clamp = T_free

# ── helpers ───────────────────────────────────────────────────────────────
def fit_pca(traj_a, traj_b):
    """Fit PCA basis on two trajectories; return a project() fn and n_components."""
    combined = np.concatenate([traj_a, traj_b], axis=0)
    mean     = combined.mean(axis=0)
    n_dim    = combined.shape[1]
    k        = min(2, n_dim)
    if n_dim == 1:
        project = lambda t: np.column_stack([t - mean, np.zeros(len(t))])
        return project, k
    _, _, Vt   = np.linalg.svd(combined - mean, full_matrices=False)
    components = Vt[:k]
    project    = lambda t: (np.array(t) - mean) @ components.T
    return project, k

def gradient_lc(ax, proj, cmap_name):
    """Draw a time-coloured LineCollection on ax; return handle for legend."""
    segs   = np.concatenate([proj[:-1, np.newaxis], proj[1:, np.newaxis]], axis=1)
    cmap   = plt.get_cmap(cmap_name)
    colors = cmap(0.35 + 0.65 * np.linspace(0, 1, len(segs)))
    lc     = LineCollection(segs, colors=colors, lw=1.8, zorder=2)
    ax.add_collection(lc)
    handle, = ax.plot([], [], color=cmap(0.7), lw=1.8)   # legend proxy
    ax.scatter(proj[0, 0],  proj[0, 1],  marker="o", s=50,
               color=cmap(0.4), zorder=5, edgecolors="k", lw=0.4)
    ax.scatter(proj[-1, 0], proj[-1, 1], marker="+", s=100,
               color=cmap(0.9), zorder=5, lw=1.5)
    return handle

# ── Plot 1 – per-neuron time series (rows = noise, cols = neurons) ────────
for l in range(N_LAYERS):
    n_neurons = sweep_pcn_full[0][l].shape[1]
    fig, axes = plt.subplots(
        N_NOISE, n_neurons,
        figsize=(max(2.8 * n_neurons, 4), 2.5 * N_NOISE),
        sharex=True, sharey="row",
        squeeze=False,
    )
    fig.suptitle(layer_labels[l], fontsize=13, fontweight="bold")

    for ni, mag in enumerate(NOISE_MAGS):
        pcn_acts    = sweep_pcn_full[ni][l]
        approx_acts = sweep_approx_full[ni][l]
        rclamp_val  = float(fp_acts[-1][0]) + float(noise_dir[0]) * mag

        for i in range(n_neurons):
            ax = axes[ni, i]
            ax.plot(t_axis, pcn_acts[:, i],    color="steelblue",  lw=1.2, label="PCN")
            ax.plot(t_axis, approx_acts[:, i], color="darkorange", lw=1.2, ls="--", label="approx-PC")
            ax.axvline(t_clamp, color="gray", ls=":", lw=0.8)
            if l == N_LAYERS - 1:   # output layer: show target
                ax.axhline(rclamp_val, color="crimson", ls="--", lw=0.9, alpha=0.8)
            if ni == 0:
                ax.set_title(f"neuron {i}", fontsize=8)
            if i == 0:
                ax.set_ylabel(f"noise\n{mag:.2f}", fontsize=7)
            if ni == N_NOISE - 1:
                ax.set_xlabel("timestep", fontsize=7)
            if ni == 0 and i == n_neurons - 1:
                ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    out = f"dynamics_layer{l + 1}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)

# ── Plot 2 – PCA (rows = noise, cols = layers) ───────────────────────────
# Fit one PCA basis per layer using the first noise level; reuse for all rows
# so trajectories share the same axes and the fixed point is always at the origin.
pca_fns = [
    fit_pca(sweep_pcn_clamp[0][l], sweep_approx_clamp[0][l])
    for l in range(N_LAYERS)
]

# Compute shared axis limits per layer across all noise levels
pca_limits = []
for l in range(N_LAYERS):
    project, _ = pca_fns[l]
    all_pts = np.concatenate([
        project(sweep_pcn_clamp[ni][l])[:, :2]
        for ni in range(N_NOISE)
    ] + [
        project(sweep_approx_clamp[ni][l])[:, :2]
        for ni in range(N_NOISE)
    ], axis=0)
    pad = (all_pts.max() - all_pts.min()) * 0.05 + 1e-6
    pca_limits.append((all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad,
                       all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad))

fig_pca, axes_pca = plt.subplots(
    N_NOISE, N_LAYERS,
    figsize=(4.5 * N_LAYERS, 4.0 * N_NOISE),
    squeeze=False,
)
fig_pca.suptitle("Clamped-phase dynamics — PC1 vs PC2", fontsize=13, fontweight="bold")

for ni, mag in enumerate(NOISE_MAGS):
    for l in range(N_LAYERS):
        ax = axes_pca[ni, l]
        project, k = pca_fns[l]
        pcn_p    = project(sweep_pcn_clamp[ni][l])[:, :2]
        approx_p = project(sweep_approx_clamp[ni][l])[:, :2]
        h_pcn    = gradient_lc(ax, pcn_p,    "Blues")
        h_approx = gradient_lc(ax, approx_p, "Oranges")
        xmin, xmax, ymin, ymax = pca_limits[l]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("PC 1", fontsize=7)
        if ni == 0:
            ax.set_title(layer_labels[l], fontsize=9)
        if l == 0:
            ax.set_ylabel(f"noise={mag:.2f}\nPC 2", fontsize=7)
        else:
            ax.set_ylabel("PC 2" if k >= 2 else "(1-D)", fontsize=7)
        if ni == 0 and l == N_LAYERS - 1:
            ax.legend([h_pcn, h_approx], ["PCN", "approx-PC"], fontsize=7)

plt.tight_layout()
out_pca = "dynamics_pca.png"
fig_pca.savefig(out_pca, dpi=150)
print(f"Saved {out_pca}")
plt.close(fig_pca)

# ── Plot 3 – heatmap: layer × noise ──────────────────────────────────────
fig_hm, ax_hm = plt.subplots(figsize=(max(3, 1.2 * N_LAYERS), max(3, 1.0 * N_NOISE)))

im = ax_hm.imshow(
    diff_matrix,
    aspect="auto",
    cmap="viridis",
    origin="lower",
)
fig_hm.colorbar(im, ax=ax_hm, label="‖PCN − approx-PC‖")

ax_hm.set_xticks(range(N_LAYERS))
ax_hm.set_xticklabels([f"L{l+1}" for l in range(N_LAYERS)], fontsize=9)
ax_hm.set_yticks(range(N_NOISE))
ax_hm.set_yticklabels([f"{m:.2f}" for m in NOISE_MAGS], fontsize=9)
ax_hm.set_xlabel("Layer", fontsize=10)
ax_hm.set_ylabel("Noise magnitude", fontsize=10)
ax_hm.set_title("Final activation error: PCN vs approx-PC", fontsize=11, fontweight="bold")

# annotate cells
for ni in range(N_NOISE):
    for l in range(N_LAYERS):
        ax_hm.text(l, ni, f"{diff_matrix[ni, l]:.3f}",
                   ha="center", va="center", fontsize=7,
                   color="white" if diff_matrix[ni, l] > diff_matrix.max() * 0.5 else "black")

plt.tight_layout()
out_hm = "dynamics_heatmap.png"
fig_hm.savefig(out_hm, dpi=150)
print(f"Saved {out_hm}")
plt.close(fig_hm)
