import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sb
    from jax import config as jax_config
    from rma_kinetics.models.constitutive.erasable import (
        Model,
        State,
        create_tev_schedule,
    )
    from rma_kinetics.solvers import Kvaerno3
    from sensitivity import global_sensitivity

    sb.set_theme(context="talk", style="ticks", font="Arial", palette="crest")
    plt.rc("axes.spines", top=False, right=False)

    jax_config.update("jax_enable_x64", True)
    data_dir = os.path.join("notebooks", "data", "constitutive_ferma")
    return (
        Kvaerno3,
        Model,
        State,
        create_tev_schedule,
        data_dir,
        global_sensitivity,
        np,
        os,
        pl,
        plt,
        sb,
    )


@app.cell
def _(Kvaerno3, Model, State, create_tev_schedule, np):
    sim_config = {
        "t0": 0,
        "tf": 504.5,
        "dt": 0.5,
        "init_state": State(),
        "solver": Kvaerno3(),
    }

    time_grid = np.arange(
        sim_config["t0"],
        sim_config["tf"] + 0.5 * sim_config["dt"],
        sim_config["dt"],
    )

    tev_schedule = create_tev_schedule(
        11.4285714286, start_time=168, repeat=1, interval=336
    )

    def map_model(params):
        model = Model(tev_schedule, *params)
        solution = model.solve(**sim_config)
        return np.interp(time_grid, solution.ts, solution.plasma_rma)

    return map_model, time_grid


@app.cell
def _(np):
    range = np.array([-0.5, 0.5])
    params = [0.4, 0.6, 0.007, 0.0015, 180, 0.5]
    param_space = {
        "num_vars": len(params),
        "names": [
            "rma_prod_rate",
            "rma_rt_rate",
            "rma_deg_rate",
            "tev_vd",
            "tev_deg",
            "tev_cut",
        ],
        "bounds": [p * (1 + range) for p in params],
        "outputs": "Y",
    }
    return (param_space,)


@app.cell
def _(global_sensitivity, map_model, np, param_space, time_grid):
    morris_y, morris_sens = global_sensitivity(map_model, param_space, 250)
    time = time_grid
    y_mean = np.mean(morris_y, axis=0)
    mu_star = np.array([s["mu_star"] for s in morris_sens])
    mu_conf = np.array([s["mu_star_conf"] for s in morris_sens])
    sigma = np.array([s["sigma"] for s in morris_sens])
    return mu_conf, mu_star, sigma, time


@app.cell
def _(data_dir, mu_conf, mu_star, os, plt, time):
    param_labels = [
        "$k_{RMA}$",
        "$k_{RT}$",
        "$\\gamma_{RMA}$",
    ]
    linestyles = ["-", ":", "--"]

    for _i, _label in enumerate(param_labels):
        _mu_star = mu_star[:, _i]
        _mu_conf = mu_conf[:, _i]
        plt.plot(time, _mu_star, label=_label, linestyle=linestyles[_i])
        plt.fill_between(time, _mu_star - _mu_conf, _mu_star + _mu_conf, alpha=0.25)

    plt.xlabel("Time (hr)")
    plt.ylabel("Mean Morris Sensitivity, $µ^*$")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean.svg"))
    plt.gca()
    return linestyles, param_labels


@app.cell
def _(mu_star):
    mu_star
    return


@app.cell
def _(mu_conf, mu_star, np, pl, sigma):
    # sens at selected time points (summary)
    timepoints = [336, 337, 1008, 1009]
    selected_params = ["Production", "Degradation"]
    selected_idx = [0, 2]

    _time = []
    _params = []
    mu = []
    conf = []
    norm_sigma = []

    for t in timepoints:
        mu_t = mu_star[t, selected_idx]
        max_mu = np.max(mu_t)
        if max_mu == 0:
            norm_mu = np.zeros_like(mu_t)
            norm_conf = np.zeros_like(mu_t)
        else:
            norm_mu = mu_t / max_mu
            norm_conf = mu_conf[t, selected_idx] / max_mu

        sigma_t = sigma[t, selected_idx]
        max_sigma = np.max(sigma_t)
        if max_sigma == 0:
            norm_sigma_t = np.zeros_like(sigma_t)
        else:
            norm_sigma_t = sigma_t / max_sigma

        _params.extend(selected_params)
        _time.extend([t] * len(selected_params))
        mu.extend(norm_mu)
        conf.extend(norm_conf)
        norm_sigma.extend(norm_sigma_t)

    mu_df = pl.DataFrame(
        {
            "time": _time,
            "params": _params,
            "mu_norm": mu,
            "conf_norm": conf,
            "sigma_norm": norm_sigma,
        }
    )
    return mu_df, selected_params


@app.cell
def _(data_dir, mu_df, np, os, pl, plt, sb, selected_params):
    # sensitivity summary around TEV dose times
    time_pairs = [
        (336, 337, "1 week"),
        (1008, 1009, "3 week"),
    ]
    x = np.arange(len(selected_params))
    width = 0.36
    colors = sb.color_palette("crest", n_colors=4)

    for t_minus, t_plus, subtitle in time_pairs:
        fig, ax = plt.subplots()
        minus = mu_df.filter(pl.col("time") == t_minus)
        plus = mu_df.filter(pl.col("time") == t_plus)

        y_minus = [
            minus.filter(pl.col("params") == p)["mu_norm"][0] for p in selected_params
        ]
        e_minus = [
            minus.filter(pl.col("params") == p)["conf_norm"][0] for p in selected_params
        ]
        y_plus = [
            plus.filter(pl.col("params") == p)["mu_norm"][0] for p in selected_params
        ]
        e_plus = [
            plus.filter(pl.col("params") == p)["conf_norm"][0] for p in selected_params
        ]

        ax.bar(
            x - width / 2,
            y_minus,
            width,
            label="-Tev",
            yerr=e_minus,
            color="darkgrey",
            alpha=0.5,
        )
        ax.bar(
            x + width / 2,
            y_plus,
            width,
            label="+Tev",
            yerr=e_plus,
            color="#038FA7",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(selected_params)
        # ax.set_title(f"{subtitle}")
        ax.set_ylabel("Relative Importance")
        ax.legend(frameon=False, loc="upper right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(data_dir, f"norm_importance_{subtitle.replace(' ', '_')}.svg")
        )
        plt.show()
    return time_pairs, width, x


@app.cell
def _(data_dir, linestyles, os, param_labels, plt, sigma, time):
    for _i, _label in enumerate(param_labels):
        _sigma = sigma[:, _i]
        plt.plot(time, sigma, linestyle=linestyles[_i])

    plt.xlabel("Time (hr)")
    plt.ylabel("Std. Morris Sensitivity, $\\sigma$")
    plt.legend(["$k_{RMA}$", "$k_{RT}$", "$\\gamma_{RMA}$"], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, mu_df, os, pl, plt, selected_params, time_pairs, width, x):
    for _t_minus, _t_plus, _subtitle in time_pairs:
        _fig, _ax = plt.subplots(figsize=(6.4, 5.5))
        _minus = mu_df.filter(pl.col("time") == _t_minus)
        _plus = mu_df.filter(pl.col("time") == _t_plus)

        _y_minus = [
            _minus.filter(pl.col("params") == p)["sigma_norm"][0]
            for p in selected_params
        ]

        _y_plus = [
            _plus.filter(pl.col("params") == p)["sigma_norm"][0]
            for p in selected_params
        ]

        _ax.bar(
            x - width / 2,
            _y_minus,
            width,
            label="-Tev",
            color="darkgrey",
            alpha=0.5,
        )
        _ax.bar(
            x + width / 2,
            _y_plus,
            width,
            label="+Tev",
            color="#038FA7",
        )

        _ax.set_xticks(x)
        _ax.set_xticklabels(selected_params)
        # ax.set_title(f"{subtitle}")
        _ax.set_ylabel("Relative Nonlinearity or Interaction")
        # _ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                data_dir, f"norm_interaction_{_subtitle.replace(' ', '_')}.svg"
            )
        )
        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
