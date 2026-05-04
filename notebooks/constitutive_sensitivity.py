import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.constitutive import Model, State
    from rma_kinetics.solvers import Dopri5
    from sensitivity import global_sensitivity
    from jax import config as jax_config

    import numpy as np
    import polars as pl
    import seaborn as sb
    import matplotlib.pyplot as plt
    import os

    sb.set_theme(context="talk", style="ticks", font="Arial", palette="crest")
    plt.rc("axes.spines", top=False, right=False)

    jax_config.update("jax_enable_x64", True)
    data_dir = os.path.join("notebooks", "data", "aav_rma_timecourse")
    return (
        Dopri5,
        Model,
        State,
        data_dir,
        global_sensitivity,
        np,
        os,
        pl,
        plt,
        sb,
    )


@app.cell
def _(Dopri5, Model, State):
    sim_config = {
        "t0": 0,
        "tf": 504,
        "dt": 1,
        "init_state": State(),
        "solver": Dopri5()
    }

    def map_model(params):
        model = Model(*params)
        solution = model.solve(**sim_config)
        return solution.plasma_rma

    return map_model, sim_config


@app.cell
def _(np):
    range = np.array([-0.5, 0.5])
    params = [0.2, 0.6, 0.007]
    param_space = {
        "num_vars": 3,
        "names": ["rma_prod_rate", "rma_rt_rate", "rma_deg_rate"],
        "bounds": [p * (1 + range) for p in params],
        "outputs": "Y"
    }
    return (param_space,)


@app.cell
def _(global_sensitivity, map_model, np, param_space, sim_config):
    morris_y, morris_sens = global_sensitivity(map_model, param_space, 250)
    time = np.linspace(sim_config["t0"], sim_config["tf"]+1, sim_config["tf"]+1)
    y_mean = np.mean(morris_y, axis=0)
    mu_star = np.array([s['mu_star'] for s in morris_sens])
    mu_conf = np.array([s['mu_star_conf'] for s in morris_sens])
    sigma = np.array([s['sigma'] for s in morris_sens])
    return mu_conf, mu_star, sigma, time


@app.cell
def _(data_dir, mu_conf, mu_star, os, plt, time):
    param_labels = ["$k_{RMA}$", "$k_{RT}$", "$\\gamma_{RMA}$"]
    linestyles = ["-", ":", "--"]

    for _i, _label in enumerate(param_labels):
        _mu_star = mu_star[:,_i]
        _mu_conf = mu_conf[:,_i]
        plt.plot(time, _mu_star, label=_label, linestyle=linestyles[_i])
        plt.fill_between(
            time,
            _mu_star - _mu_conf,
            _mu_star + _mu_conf,
            alpha=0.25
        )

    plt.xlabel("Time (hr)")
    plt.ylabel("Mean Morris Sensitivity, $µ^*$")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean.svg"))
    plt.gca()
    return linestyles, param_labels


@app.cell
def _(data_dir, linestyles, os, param_labels, plt, sigma, time):
    for _i, _label in enumerate(param_labels):
        plt.plot(time, sigma[:, _i], label=_label, linestyle=linestyles[_i])

    plt.xlabel("Time (hr)")
    plt.ylabel("Std. Morris Sensitivity, $\\sigma$")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std.svg"))
    plt.gca()
    return


@app.cell
def _(mu_conf, mu_star, np, pl, sigma):
    # sens std at selected time points (summary)
    timepoints = [12, 252, 504]
    _time = []
    _params = []
    mu = []
    conf = []
    norm_sigma = []
    for t in timepoints:
        mu_t = mu_star[t]
        max_mu = np.max(mu_t)
        norm_mu = mu_t / max_mu
        norm_conf = mu_conf[t] / max_mu

        sigma_t = sigma[t]
        norm_sigma_t = sigma_t / np.max(sigma_t)

        _params.extend(["Production", "BBB Transport", "Degradation"])
        _time.extend([t]*3)
        mu.extend(norm_mu)
        conf.extend(norm_conf)
        norm_sigma.extend(norm_sigma_t)

    mu_df = pl.DataFrame({ "time": _time, "params": _params, "mu_norm": mu, "conf_norm": conf, "sigma_norm": norm_sigma})
    return mu_df, t


@app.cell
def _(data_dir, mu_df, np, os, pl, plt, sb, t):
    # sensitivity at at 1, 2, and 3 week timepoints (summary)
    _params = ["Production", "BBB Transport", "Degradation"]
    times = sorted(mu_df["time"].unique())
    colors = sb.color_palette("crest", n_colors=6)
    alphas = [0.5, 1, 1]

    x = np.arange(len(_params))
    width = 0.2

    # fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    fig, ax = plt.subplots()

    for i, _t in enumerate(times):
        subset = mu_df.filter(pl.col("time") == _t)

        y = subset["mu_norm"]
        yerr = subset["conf_norm"]
        #s = subset["sigma_norm"]

        plt.bar(
            x + i * width,
            y,
            width,
            label=str(t),
            yerr=yerr,
            color=colors[i*2],
            alpha=alphas[i]
        )

        """
        ax[1].bar(
            x + i * width,
            s,
            width,
            label=str(t),
            color=colors[i*2],
            alpha=alphas[i]
        )
        """

    ax.set_xticks(x + width * (len(times) - 1) / 2)
    ax.set_xticklabels(_params)
    ax.set_ylabel("Relative Importance")

    #ax[1].set_xticks(x + width * (len(times) - 1) / 2)
    #ax[1].set_xticklabels(_params)
    #ax[1].set_ylabel("Relative Nonlinearity or Interaction")

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_importance.svg"))
    plt.show()
    return alphas, colors, times, width, x


@app.cell
def _(alphas, colors, data_dir, mu_df, os, pl, plt, t, times, width, x):
    # sensitivity at at 1, 2, and 3 week timepoints (summary)
    _params = ["Production", "BBB Transport", "Degradation"]
    #times = sorted(mu_df["time"].unique())
    #colors = sb.color_palette("crest", n_colors=6)
    #alphas = [0.5, 1, 1]

    #x = np.arange(len(_params))
    #width = 0.2

    # fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    _fig, _ax = plt.subplots()

    for _i, _t in enumerate(times):
        _subset = mu_df.filter(pl.col("time") == _t)
        s = _subset["sigma_norm"]
    
        plt.bar(
            x + _i * width,
            s,
            width,
            label=str(t),
            color=colors[_i*2],
            alpha=alphas[_i]
        )

    _ax.set_xticks(x + width * (len(times) - 1) / 2)
    _ax.set_xticklabels(_params)
    _ax.set_ylabel("Relative Nonlinearity or Interaction")

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_interaction.svg"))
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
