import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sb
    from jax import config as jax_config
    from rma_kinetics.models import cno
    from rma_kinetics.models.chemogenetic import Model, State
    from rma_kinetics.models.dox import AccessPeriod, Model as DoxModel
    from rma_kinetics.solvers import Kvaerno3
    from sensitivity import global_sensitivity

    sb.set_theme(context="talk", style="ticks")
    plt.rc("axes.spines", top=False, right=False)

    jax_config.update("jax_enable_x64", True)
    data_dir = os.path.join("notebooks", "data", "chemogenetic")
    return (
        AccessPeriod,
        DoxModel,
        Kvaerno3,
        Model,
        State,
        cno,
        data_dir,
        global_sensitivity,
        np,
        os,
        plt,
        sb,
    )


@app.cell
def _(AccessPeriod, DoxModel, cno):
    mouse_weight = 0.03

    dox_schedule = [AccessPeriod(40, 0, 48)]
    dox_pk_model = DoxModel(schedule=dox_schedule)

    cno_dose_cls = cno.CnoDose
    cno_doses = [cno_dose_cls(mouse_weight, 48)]
    cno_pk_model = cno.Model(doses=cno_doses)
    return cno_pk_model, dox_pk_model


@app.cell
def _(State):
    dox_intake_rate = 1.875e-4 * 0.9 * 34.8 / (444.4 * 0.21) * 1e6
    plasma_dox_ss = 0.8 * dox_intake_rate / 0.2
    brain_dox_ss = 0.2 * plasma_dox_ss

    init_state = State(brain_dox=brain_dox_ss, plasma_dox=plasma_dox_ss)
    return


@app.cell
def _(np, os):
    base_params = np.array(
        [
            0.35,
            0.8,
            0.007,
            5.0,
            10.0,
            0.05,
            5.0,
            5.0,
            3.0,
            10.0,
            5.0,
            0.002,
            0.1,
        ]
    )

    params_path = os.path.join("notebooks", "data", "chemogenetic", "cno_1_params.npy")
    if os.path.exists(params_path):
        sa_params = np.load(params_path)
    else:
        sa_params = base_params
    return (sa_params,)


@app.cell
def _(Kvaerno3, Model, State, cno_pk_model, dox_pk_model, np):
    sim_config = {
        "t0": 0,
        "tf": 96,
        "dt": 1,
        "solver": Kvaerno3(),
    }

    time_grid = np.arange(
        sim_config["t0"],
        sim_config["tf"] + sim_config["dt"],
        sim_config["dt"],
    )

    def map_model(params):
        model = Model(
            rma_prod=params[0],
            rma_bbb_transport=params[1],
            rma_deg=params[2],
            dox_pk_model=dox_pk_model,
            dox_tta_kd=params[3],
            tta_prod=params[4],
            tta_deg=params[5],
            tta_kd=params[6],
            cno_pk_model=cno_pk_model,
            cno_ec50=params[7],
            clz_ec50=params[8],
            dreadd_prod=params[9],
            dreadd_deg=1,
            dreadd_ec50=params[10],
            leaky_rma_prod=params[11],
            leaky_tta_prod=params[12],
        )

        _init_state = State(brain_dox=0, plasma_dox=0, dreadd=params[9])
        solution = model.solve(**sim_config, init_state=_init_state)
        return np.interp(time_grid, solution.ts, solution.plasma_rma)

    return map_model, time_grid


@app.cell
def _(np, sa_params):
    range = np.array([-0.5, 0.5])
    param_space = {
        "num_vars": 13,
        "names": [
            "rma_prod",
            "rma_bbb_transport",
            "rma_deg",
            "dox_tta_kd",
            "tta_prod",
            "tta_deg",
            "tta_kd",
            "cno_ec50",
            "clz_ec50",
            "dreadd_prod",
            "dreadd_ec50",
            "leaky_rma_prod",
            "leaky_tta_prod",
        ],
        "bounds": [p * (1 + range) for p in sa_params],
        "outputs": "Y",
    }
    return (param_space,)


@app.cell
def _(global_sensitivity, map_model, np, param_space, time_grid):
    morris_y, morris_sens = global_sensitivity(map_model, param_space, 250)
    sens_time = time_grid
    mu_star = np.array([s["mu_star"] for s in morris_sens])
    mu_conf = np.array([s["mu_star_conf"] for s in morris_sens])
    sigma = np.array([s["sigma"] for s in morris_sens])
    return mu_conf, mu_star, sens_time, sigma


@app.cell
def _(sb):
    sb.set_theme(context="talk", style="ticks")
    return


@app.cell
def _(data_dir, mu_conf, mu_star, np, os, plt, sb):
    idx = np.arange(13)
    sa_labels = [
        "$k_{RMA}$",
        "$k_{RT}$",
        "$\\gamma_{RMA}$",
        "$K_{D_{Dox}}$",
        "$k_{tTA}$",
        "$\\gamma_{tTA}$",
        "$K_{D_{tTA}}$",
        "$EC_{50_{CNO}}$",
        "$EC_{50_{CLZ}}$",
        "$[Dq]_{ss}$",
        "$EC_{50_{Dq}}$",
        "$k_{0_{RMA}}$",
        "$k_{0_{tTA}}$",
    ]

    _fig, _ax = plt.subplots()
    _ax.bar(
        sa_labels,
        [mu_star[-1, i] for i in idx],
        yerr=[mu_conf[-1, j] for j in idx],
        color="lightgrey",
    )

    plt.ylabel("Mean Morris Sensitivity, $µ^*$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean_tf.svg"))
    plt.gca()
    return idx, sa_labels


@app.cell
def _(data_dir, idx, mu_conf, mu_star, np, os, plt, sa_labels, sb):
    _fig, _ax = plt.subplots()
    max_mu_star = np.max(mu_star[-1, :])
    _ax.bar(
        sa_labels,
        [mu_star[-1, i] / max_mu_star for i in idx],
        yerr=[mu_conf[-1, j] / max_mu_star for j in idx],
        color="lightgrey",
    )

    plt.ylabel("Relative Ranking")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    _fig.set_figheight(5.2)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_tf.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, mu_conf, mu_star, os, plt, sa_labels, sb, sens_time):
    top_params = mu_star[-1, :].argsort()[-5:][::-1]
    _fig, _ax = plt.subplots()

    for _i in top_params:
        _ax.plot(sens_time, mu_star[:, _i], label=sa_labels[_i])
        _ax.fill_between(
            sens_time,
            mu_star[:, _i] - mu_conf[:, _i],
            mu_star[:, _i] + mu_conf[:, _i],
            alpha=0.25,
        )

    plt.ylabel("Mean Morris Sensitivity, $µ^*$")
    plt.xlabel("Time (hr)")
    plt.legend(frameon=False)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean.svg"))
    plt.gca()
    return (top_params,)


@app.cell
def _(data_dir, idx, os, plt, sa_labels, sb, sigma):
    _fig, _ax = plt.subplots()
    _ax.bar(
        sa_labels,
        [sigma[-1, i] for i in idx],
        color="lightgrey",
    )

    plt.ylabel("Std. Morris Sensitivity, $\\sigma$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std_tf.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, idx, np, os, plt, sa_labels, sb, sigma):
    _fig, _ax = plt.subplots()
    max_sigma = np.max(sigma[-1, :])
    _ax.bar(
        sa_labels,
        [sigma[-1, i] / max_sigma for i in idx],
        color="lightgrey",
    )

    plt.ylabel("Relative Nonlinearity\nor Interaction")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    _fig.set_figheight(5.2)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_tf.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, os, plt, sa_labels, sb, sens_time, sigma, top_params):
    _fig, _ax = plt.subplots()
    for _i in top_params:
        _ax.plot(sens_time, sigma[:, _i], label=sa_labels[_i])

    plt.ylabel("Std. Morris Sensitivity, $\\sigma$")
    plt.xlabel("Time (hr)")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std.svg"))
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
