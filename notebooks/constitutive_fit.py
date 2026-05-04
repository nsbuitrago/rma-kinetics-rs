import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sb
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import wrap_py
    from rma_kinetics.models.constitutive import Model, State
    from rma_kinetics.solvers import Dopri5
    from utils import rlu_to_nm

    sb.set_theme(context="notebook", style="ticks", font="Arial")
    plt.rc("axes.spines", top=False, right=False)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Constitutive RMA Kinetic Model

    This analysis notebook is for the constitutive released markers of activity kinetic model. A simple model is developed and fit to RMA timecourse data measured over the course of 3 weeks.

    Data for RMA expression in the CA1 region of the hippocampus, caudate putamen, and substantia nigra is available. Select the brain region below to use the corresponding data in the notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_id = mo.ui.radio(
        options=["CA1", "CP", "SN"], value="CA1", label="RMA Dataset"
    )
    dataset_id
    return (dataset_id,)


@app.cell(hide_code=True)
def _(dataset_id):
    data_dir = os.path.join(
        "notebooks", "data", "aav_rma_timecourse", dataset_id.value
    )
    return (data_dir,)


@app.function(hide_code=True)
def get_df(dataset_id: str, data_dir: str):
    raw_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    df = rlu_to_nm(raw_df)

    summary_df = (
        df.group_by("time")
        .agg(
            [
                pl.col("concentration").mean().alias("mean"),
                pl.col("concentration").std().alias("std"),
            ]
        )
        .sort("time")
    )

    return df, summary_df


@app.cell(hide_code=True)
def _(data_dir, dataset_id):
    df, summary_df = get_df(dataset_id.value, data_dir)
    df_plot = df.unpivot(
        on=["rlu", "concentration"],
        index="time",
        variable_name="output_type",
        value_name="value",
    )

    grid = sb.FacetGrid(data=df_plot, col="output_type", sharey=False)
    grid.map(sb.pointplot, "time", "value", order=[0, 336, 504])

    grid.axes[0][0].set_yscale("log")
    grid.axes[0][0].set_ylabel("RLU (a.u.)")
    grid.axes[0][0].set_xlabel("Time (hr)")
    grid.axes[0][0].set_title("")

    grid.axes[0][1].set_ylabel("Concentration (nM)")
    grid.axes[0][1].set_xlabel("Time (hr)")
    grid.axes[0][1].set_title("")

    grid.fig.suptitle(f"AAV RMA timecourse - {dataset_id.value}")
    plt.tight_layout()
    plt.gcf()
    return df, summary_df


@app.cell
def _(df):
    fit_df = (
        df.select(["mouse_id", "time", "concentration"])
        .with_columns(
            [
                pl.col("mouse_id").cast(pl.Utf8),
                pl.col("time").cast(pl.Int64),
                pl.col("concentration").cast(pl.Float64),
            ]
        )
        .sort(["mouse_id", "time"])
    )

    obs_plasma_rma = fit_df["concentration"].to_numpy().astype(float)
    obs_time = fit_df["time"].to_numpy().astype(int)
    n_obs = obs_plasma_rma.size
    mouse_id = fit_df["mouse_id"].to_numpy().astype(int)
    n_mice = fit_df.group_by(pl.col("mouse_id")).n_unique().height
    tf = float(np.max(obs_time))

    fit_df
    return fit_df, mouse_id, n_mice, n_obs, obs_plasma_rma, obs_time, tf


@app.cell
def _():
    return


@app.cell
def _(mouse_id, n_mice, n_obs, obs_plasma_rma, obs_time, tf):
    @wrap_py(
        itypes=[pt.dvector, pt.dscalar, pt.dscalar], otypes=[pt.dvector]
    )
    def mechanistic_expectation(log_prod_mouse, log_bbb, log_deg):
        """
        Inference types are
        prod rate: vector of production rates (1 per mouse)
        bbb transport rate: scalar for population bbb transport
        deg rate: scalar for population degradation rate
        """
        prod_mouse = np.exp(np.asarray(log_prod_mouse, dtype=float))
        bbb_transport = float(np.exp(log_bbb))
        deg = float(np.exp(log_deg))

        pred = np.empty(n_obs, dtype=float)

        for m in range(n_mice):
            mask = mouse_id == m

            if not np.any(mask):
                continue

            model = Model(prod_mouse[m], bbb_transport, deg)
            try:
                solution = model.solve(0.0, tf, 1.0, State(), Dopri5())
                pred[mask] = solution.plasma_rma[obs_time[mask]]
            except:
                continue


        return pred


    coords = {
        "mouse": np.arange(n_mice),
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(coords=coords) as mixed_effect_model:
        mean_log_prod = pm.Normal("mu_log_prod", mu=np.log(0.2), sigma=0.35)
        log_prod_mouse = pm.Normal(
            "log_prod_mouse",
            mu=mean_log_prod,
            sigma=0.5,
            dims="mouse",
        )

        # population-level priors (log-normal parameterization)
        log_bbb = pm.Normal("log_bbb", mu=np.log(0.6), sigma=0.15)
        log_deg = pm.Normal("log_deg", mu=np.log(0.007), sigma=0.36)
        var_obs = pm.HalfNormal("sigma_obs", sigma=0.3)

        mean_plasma_rma = pm.Deterministic(
            "mu",
            mechanistic_expectation(log_prod_mouse, log_bbb, log_deg),
            dims="obs_id",
        )

        pm.Normal(
            "y",
            mu=mean_plasma_rma,
            sigma=var_obs,
            observed=obs_plasma_rma,
            dims="obs_id",
        )

        idata = pm.sample(
            draws=10000,
            tune=10000,
            chains=6,
            cores=4,
            random_seed=42,
            step=pm.DEMetropolisZ(),
            return_inferencedata=True,
        )

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y"],
            random_seed=42,
            return_inferencedata=True,
        )

    return idata, ppc


@app.cell
def _(idata):
    summary = az.summary(
        idata,
        var_names=[
            "mu_log_prod",
            "log_bbb",
            "log_deg",
        ],
        round_to=4,
    )
    summary
    return (summary,)


@app.cell
def _(data_dir, summary):
    summary.to_csv(os.path.join(data_dir, "parameter_fit_summary.csv"))
    return


@app.cell
def _(idata):
    bbb_samples = np.exp(idata.posterior["log_bbb"].values.reshape(-1))
    deg_samples = np.exp(idata.posterior["log_deg"].values.reshape(-1))
    bbb_q = np.quantile(bbb_samples, [0.05, 0.5, 0.95])
    deg_q = np.quantile(deg_samples, [0.05, 0.5, 0.95])
    mo.md(
        f"""
        ### Population Parameters (Original Scale)
        - `bbb_transport` median [90% interval]: {bbb_q[1]:.4f} [{bbb_q[0]:.4f}, {bbb_q[2]:.4f}]
        - `deg` median [90% interval]: {deg_q[1]:.5f} [{deg_q[0]:.5f}, {deg_q[2]:.5f}]
        """
    )
    return


@app.cell
def _(data_dir, idata):
    az.plot_trace(
        idata,
        var_names=[
            "mu_log_prod",
            "log_bbb",
            "log_deg",
        ],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "mmc_trace.svg"))
    plt.gcf()
    return


@app.cell
def _(fit_df, mouse_id, n_mice, obs_plasma_rma, obs_time, ppc):
    y_ppc = np.asarray(ppc.posterior_predictive["y"], dtype=float)
    y_ppc_samples = y_ppc.reshape(-1, y_ppc.shape[-1])
    y_mean = y_ppc_samples.mean(axis=0)
    y_hdi = az.hdi(y_ppc_samples, hdi_prob=0.9)
    mouse_labels = fit_df["mouse_id"].unique().sort().to_list()
    fig, axes = plt.subplots(1, n_mice, figsize=(4 * n_mice, 3), sharey=True)
    if n_mice == 1:
        axes = [axes]
    for m in range(n_mice):
        ax = axes[m]
        mask = mouse_id == m
        order = np.argsort(obs_time[mask])
        t = obs_time[mask][order]
        y = obs_plasma_rma[mask][order]
        _mu = y_mean[mask][order]
        lo = y_hdi[mask, 0][order]
        hi = y_hdi[mask, 1][order]
        ax.fill_between(t, lo, hi, color="tab:blue", alpha=0.2, label="90% HDI")
        ax.plot(t, _mu, color="tab:blue", lw=2, label="Posterior mean")
        ax.scatter(t, y, color="black", s=30, zorder=3, label="Observed")
        ax.set_title(f"Mouse {mouse_labels[m]}")
        ax.set_xlabel("Time (hr)")
        if m == 0:
            ax.set_ylabel("Concentration (nM)")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    #plt.savefig(data_dir + "/per_mouse_posterior_mean.svg")
    plt.gcf()
    return


@app.cell
def _(idata, n_mice):
    log_prod_samples = idata.posterior["log_prod_mouse"].values
    log_bbb_samples = idata.posterior["log_bbb"].values
    log_deg_samples = idata.posterior["log_deg"].values

    def plasma_rma_fit(prod_samples, bbb_samples, deg_samples, n_mice) -> (np.typing.NDarray, np.typing.NDArray):
        log_prod = prod_samples.reshape(-1, n_mice)
        log_bbb = bbb_samples.reshape(-1)
        log_deg = deg_samples.reshape(-1)

        total_draws = log_prod.shape[0]

        trajectories = []

        for i in range(total_draws):
            mouse_i_plasma_rma = []
            bbb = np.exp(log_bbb[i])
            deg = np.exp(log_deg[i])

            for mouse in range(n_mice):
                prod = np.exp(log_prod[i, mouse])
                model = Model(prod, bbb, deg)
                solution = model.solve(0, 504, 1, State(), Dopri5())
                mouse_i_plasma_rma.append(solution.plasma_rma)

            trajectories.append(np.mean(mouse_i_plasma_rma, axis=0))

        trajectories = np.array(trajectories)
        mean_plasma_rma = trajectories.mean(axis=0)
        hdi = az.hdi(trajectories, hdi_prob=0.94)

        return mean_plasma_rma, hdi

    pop_plasma_rma, pop_plasma_rma_hdi = plasma_rma_fit(log_prod_samples, log_bbb_samples, log_deg_samples, n_mice)    
    return pop_plasma_rma, pop_plasma_rma_hdi


@app.cell
def _(pop_plasma_rma, summary_df):
    # visual check
    plt.plot(pop_plasma_rma)
    plt.errorbar(summary_df["time"], summary_df["mean"], yerr=summary_df["std"], fmt="o")
    plt.show()
    return


@app.cell
def _(data_dir, pop_plasma_rma, pop_plasma_rma_hdi):
    np.save(os.path.join(data_dir, "predicted_mean.npy"), pop_plasma_rma)
    np.save(os.path.join(data_dir, "hdi.npy"), pop_plasma_rma_hdi)
    return


if __name__ == "__main__":
    app.run()
