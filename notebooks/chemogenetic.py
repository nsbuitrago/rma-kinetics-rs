import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pytensor.tensor as pt
    import seaborn as sb
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op
    from rma_kinetics.models.chemogenetic import AdjointEngine, Model, State
    from rma_kinetics.models.cno import Dose
    from rma_kinetics.models.cno import Model as CnoModel
    from rma_kinetics.models.dox import AccessPeriod
    from rma_kinetics.models.dox import Model as DoxModel
    from rma_kinetics.solvers import Kvaerno3
    from utils import rlu_to_nm

    sb.set_theme(context="notebook", style="ticks", font="Arial")
    plt.rc("axes.spines", top=False, right=False)
    return (
        AccessPeriod,
        Apply,
        CnoModel,
        Dose,
        DoxModel,
        Kvaerno3,
        Model,
        Op,
        AdjointEngine,
        State,
        az,
        np,
        os,
        pl,
        plt,
        pm,
        pt,
        rlu_to_nm,
        sb,
    )


@app.cell
def _(AccessPeriod, CnoModel, Dose, DoxModel, Kvaerno3, Model, State):
    # setup dox model
    dox_schedule = AccessPeriod(40, 0, 24)
    dox_pk = DoxModel(schedule=[dox_schedule])

    # setup cno model
    cno_dose = Dose(0.03, 48)
    cno_pk = CnoModel(doses=[cno_dose])

    # setup chemogenetic model
    model = Model(dox_pk_model=dox_pk, cno_pk_model=cno_pk)
    # initial state
    init_state = State()
    solver = Kvaerno3()
    solution = model.solve(0, 96, 1, init_state, solver)
    return (solution,)


@app.cell
def _(plt, solution):
    plt.subplot(2, 2, 1)
    plt.plot(solution.ts, solution.plasma_rma, "k")
    plt.ylabel("Plasma RMA (nM)")
    plt.xlabel("Time (hr)")

    plt.subplot(2, 2, 2)
    plt.plot(solution.ts, solution.tta, "k")
    plt.ylabel("tTA (nM)")
    plt.xlabel("Time (hr)")

    plt.subplot(2, 2, 3)
    plt.plot(solution.ts, solution.brain_dox, "k")
    plt.ylabel("Brain Dox (nM)")
    plt.xlabel("Time (hr)")

    plt.subplot(2, 2, 4)
    plt.plot(solution.ts, solution.brain_clz, "k")
    plt.ylabel("Brain CLZ (nM)")
    plt.xlabel("Time (hr)")

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(os):
    data_dir = os.path.join("notebooks/data/chemogenetic")
    return (data_dir,)


@app.cell
def _(os, pl, rlu_to_nm):
    def get_df(data_dir: str) -> (pl.DataFrame, pl.DataFrame):
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

    return (get_df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(data_dir, get_df, pl, plt, sb):
    df, summary_df = get_df(data_dir)
    df_plot = df.filter(pl.col("cno_dose") == 1).unpivot(
        on=["rlu", "concentration"],
        index="time",
        variable_name="output_type",
        value_name="value",
    )

    grid = sb.FacetGrid(data=df_plot, col="output_type", sharey=False)
    grid.map(sb.pointplot, "time", "value", order=[0, 24, 48])

    grid.axes[0][0].set_yscale("log")
    grid.axes[0][0].set_ylabel("RLU (a.u.)")
    grid.axes[0][0].set_xlabel("Time (hr)")
    grid.axes[0][0].set_title("")

    grid.axes[0][1].set_ylabel("Concentration (nM)")
    grid.axes[0][1].set_xlabel("Time (hr)")
    grid.axes[0][1].set_title("")

    # grid.fig.suptitle(f"AAV RMA timecourse - {dataset_id.value}")
    plt.tight_layout()
    plt.gcf()
    return (df,)


@app.cell
def _(df, np, pl):
    fit_df = (
        df.filter(pl.col("cno_dose") == 1)
        .select(["mouse_id", "time", "concentration"])
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
    return fit_df, mouse_id, n_mice, n_obs, obs_plasma_rma, obs_time


@app.cell
def _(
    Apply,
    CnoModel,
    Dose,
    DoxModel,
    Op,
    AdjointEngine,
    mouse_id,
    n_mice,
    n_obs,
    np,
    obs_plasma_rma,
    obs_time,
    pm,
    pt,
):
    dox_pk_model = DoxModel()
    dox_intake_rate = 1.875e-4 * 0.9 * 34.8 / (444.4 * 0.21) * 1e6
    plasma_dox_ss = 0.8 * dox_intake_rate / 0.2
    brain_dox_ss = 0.2 * plasma_dox_ss
    pre_injection_hours = 48.0
    sim_obs_time = obs_time.astype(float) + pre_injection_hours

    cno_pk_model = CnoModel(doses=[Dose(0.03, pre_injection_hours)])
    engine = AdjointEngine(
        mouse_id=mouse_id,
        obs_time=sim_obs_time,
        n_mice=n_mice,
        dox_pk_model=dox_pk_model,
        cno_pk_model=cno_pk_model,
        plasma_dox_ss=plasma_dox_ss,
        brain_dox_ss=brain_dox_ss,
        t0=0.0,
        dt_sub=0.25,
    )
    fallback_mu = np.full(n_obs, 1e6, dtype=float)
    n_global_params = 11

    log_prod_lower = np.log(1e-4)
    log_prod_upper = np.log(1)
    log_leaky_lower = np.log(1e-6)
    log_leaky_upper = np.log(0.04)
    log_bbb_lower = np.log(1e-3)
    log_bbb_upper = np.log(5.0)
    log_deg_lower = np.log(1e-5)
    log_deg_upper = np.log(1.0)
    log_tta_prod_lower = np.log(1e-3)
    log_tta_prod_upper = np.log(100)
    log_tta_leaky_lower = np.log(1e-5)
    log_tta_leaky_upper = np.log(0.1)
    log_tta_deg_lower = np.log(1e-4)
    log_tta_deg_upper = np.log(2.0)
    log_kd_lower = np.log(1e-3)
    log_kd_upper = np.log(500.0)
    log_dreadd_prod_lower = np.log(1e-4)
    log_dreadd_prod_upper = np.log(200.0)

    class ChemogeneticVJPOp(Op):
        def __init__(self, adjoint_engine):
            self.engine = adjoint_engine

        def make_node(
            self,
            g,
            log_prod_mouse,
            log_leaky_prod_mouse,
            log_bbb,
            log_deg,
            log_tta_prod,
            log_tta_leaky_prod,
            log_tta_deg,
            log_tta_kd,
            log_dox_kd,
            log_cno_ec50,
            log_clz_ec50,
            log_dreadd_prod,
            log_dreadd_ec50,
        ):
            g = pt.as_tensor_variable(g)
            log_prod_mouse = pt.as_tensor_variable(log_prod_mouse)
            log_leaky_prod_mouse = pt.as_tensor_variable(log_leaky_prod_mouse)
            log_bbb = pt.as_tensor_variable(log_bbb)
            log_deg = pt.as_tensor_variable(log_deg)
            log_tta_prod = pt.as_tensor_variable(log_tta_prod)
            log_tta_leaky_prod = pt.as_tensor_variable(log_tta_leaky_prod)
            log_tta_deg = pt.as_tensor_variable(log_tta_deg)
            log_tta_kd = pt.as_tensor_variable(log_tta_kd)
            log_dox_kd = pt.as_tensor_variable(log_dox_kd)
            log_cno_ec50 = pt.as_tensor_variable(log_cno_ec50)
            log_clz_ec50 = pt.as_tensor_variable(log_clz_ec50)
            log_dreadd_prod = pt.as_tensor_variable(log_dreadd_prod)
            log_dreadd_ec50 = pt.as_tensor_variable(log_dreadd_ec50)

            return Apply(
                self,
                [
                    g,
                    log_prod_mouse,
                    log_leaky_prod_mouse,
                    log_bbb,
                    log_deg,
                    log_tta_prod,
                    log_tta_leaky_prod,
                    log_tta_deg,
                    log_tta_kd,
                    log_dox_kd,
                    log_cno_ec50,
                    log_clz_ec50,
                    log_dreadd_prod,
                    log_dreadd_ec50,
                ],
                [
                    log_prod_mouse.type(),
                    log_leaky_prod_mouse.type(),
                    log_bbb.type(),
                    log_deg.type(),
                    log_tta_prod.type(),
                    log_tta_leaky_prod.type(),
                    log_tta_deg.type(),
                    log_tta_kd.type(),
                    log_dox_kd.type(),
                    log_cno_ec50.type(),
                    log_clz_ec50.type(),
                    log_dreadd_prod.type(),
                    log_dreadd_ec50.type(),
                ],
            )

        def perform(self, node, inputs, outputs):
            (
                g,
                log_prod_mouse,
                log_leaky_prod_mouse,
                log_bbb,
                log_deg,
                log_tta_prod,
                log_tta_leaky_prod,
                log_tta_deg,
                log_tta_kd,
                log_dox_kd,
                log_cno_ec50,
                log_clz_ec50,
                log_dreadd_prod,
                log_dreadd_ec50,
            ) = inputs

            g = np.asarray(g, dtype=float)
            try:
                d_prod, d_leaky, d_global = self.engine.vjp(
                    g,
                    np.asarray(log_prod_mouse, dtype=float),
                    np.asarray(log_leaky_prod_mouse, dtype=float),
                    float(log_bbb),
                    float(log_deg),
                    float(log_tta_prod),
                    float(log_tta_leaky_prod),
                    float(log_tta_deg),
                    float(log_tta_kd),
                    float(log_dox_kd),
                    float(log_cno_ec50),
                    float(log_clz_ec50),
                    float(log_dreadd_prod),
                    float(log_dreadd_ec50),
                )

                outputs[0][0] = np.asarray(d_prod, dtype=float)
                outputs[1][0] = np.asarray(d_leaky, dtype=float)

                d_global = np.asarray(d_global, dtype=float)
                for i in range(d_global.size):
                    outputs[2 + i][0] = np.asarray(d_global[i], dtype=float)
            except Exception:
                outputs[0][0] = np.zeros_like(np.asarray(log_prod_mouse, dtype=float))
                outputs[1][0] = np.zeros_like(
                    np.asarray(log_leaky_prod_mouse, dtype=float)
                )
                for i in range(n_global_params):
                    outputs[2 + i][0] = np.asarray(0.0, dtype=float)

    class ChemogeneticExpectationOp(Op):
        def __init__(self, adjoint_engine):
            self.engine = adjoint_engine
            self._vjp_op = ChemogeneticVJPOp(adjoint_engine)

        def make_node(
            self,
            log_prod_mouse,
            log_leaky_prod_mouse,
            log_bbb,
            log_deg,
            log_tta_prod,
            log_tta_leaky_prod,
            log_tta_deg,
            log_tta_kd,
            log_dox_kd,
            log_cno_ec50,
            log_clz_ec50,
            log_dreadd_prod,
            log_dreadd_ec50,
        ):
            log_prod_mouse = pt.as_tensor_variable(log_prod_mouse)
            log_leaky_prod_mouse = pt.as_tensor_variable(log_leaky_prod_mouse)
            log_bbb = pt.as_tensor_variable(log_bbb)
            log_deg = pt.as_tensor_variable(log_deg)
            log_tta_prod = pt.as_tensor_variable(log_tta_prod)
            log_tta_leaky_prod = pt.as_tensor_variable(log_tta_leaky_prod)
            log_tta_deg = pt.as_tensor_variable(log_tta_deg)
            log_tta_kd = pt.as_tensor_variable(log_tta_kd)
            log_dox_kd = pt.as_tensor_variable(log_dox_kd)
            log_cno_ec50 = pt.as_tensor_variable(log_cno_ec50)
            log_clz_ec50 = pt.as_tensor_variable(log_clz_ec50)
            log_dreadd_prod = pt.as_tensor_variable(log_dreadd_prod)
            log_dreadd_ec50 = pt.as_tensor_variable(log_dreadd_ec50)

            return Apply(
                self,
                [
                    log_prod_mouse,
                    log_leaky_prod_mouse,
                    log_bbb,
                    log_deg,
                    log_tta_prod,
                    log_tta_leaky_prod,
                    log_tta_deg,
                    log_tta_kd,
                    log_dox_kd,
                    log_cno_ec50,
                    log_clz_ec50,
                    log_dreadd_prod,
                    log_dreadd_ec50,
                ],
                [pt.dvector()],
            )

        def perform(self, node, inputs, outputs):
            (
                log_prod_mouse,
                log_leaky_prod_mouse,
                log_bbb,
                log_deg,
                log_tta_prod,
                log_tta_leaky_prod,
                log_tta_deg,
                log_tta_kd,
                log_dox_kd,
                log_cno_ec50,
                log_clz_ec50,
                log_dreadd_prod,
                log_dreadd_ec50,
            ) = inputs

            try:
                mu = self.engine.predict(
                    np.asarray(log_prod_mouse, dtype=float),
                    np.asarray(log_leaky_prod_mouse, dtype=float),
                    float(log_bbb),
                    float(log_deg),
                    float(log_tta_prod),
                    float(log_tta_leaky_prod),
                    float(log_tta_deg),
                    float(log_tta_kd),
                    float(log_dox_kd),
                    float(log_cno_ec50),
                    float(log_clz_ec50),
                    float(log_dreadd_prod),
                    float(log_dreadd_ec50),
                )
                outputs[0][0] = np.asarray(mu, dtype=float)
            except Exception:
                outputs[0][0] = fallback_mu.copy()

        def grad(self, inputs, output_grads):
            g = output_grads[0]
            return list(self._vjp_op(g, *inputs))

    mechanistic_expectation = ChemogeneticExpectationOp(engine)

    coords = {
        "mouse": np.arange(n_mice),
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(coords=coords) as mixed_effect_model:
        mean_log_prod = pm.TruncatedNormal(
            "mu_log_prod",
            mu=np.log(0.2),
            sigma=0.35,
            lower=log_prod_lower,
            upper=log_prod_upper,
        )
        mean_leaky_log_prod = pm.TruncatedNormal(
            "mu_leaky_log_prod",
            mu=np.log(0.01),
            sigma=0.1,
            lower=log_leaky_lower,
            upper=log_leaky_upper,
        )

        log_prod_mouse = pm.TruncatedNormal(
            "log_prod_mouse",
            mu=mean_log_prod,
            sigma=0.3,
            lower=log_prod_lower,
            upper=log_prod_upper,
            dims="mouse",
        )

        log_leaky_prod_mouse = pm.TruncatedNormal(
            "log_leaky_prod_mouse",
            mu=mean_leaky_log_prod,
            sigma=0.1,
            lower=log_leaky_lower,
            upper=log_leaky_upper,
            dims="mouse",
        )

        # population-level priors (log-normal parameterization)
        log_bbb = pm.TruncatedNormal(
            "log_bbb",
            mu=np.log(0.6),
            sigma=0.15,
            lower=log_bbb_lower,
            upper=log_bbb_upper,
        )
        log_deg = pm.TruncatedNormal(
            "log_deg",
            mu=np.log(0.007),
            sigma=0.36,
            lower=log_deg_lower,
            upper=log_deg_upper,
        )
        log_tta_prod = pm.TruncatedNormal(
            "log_tta_prod",
            mu=np.log(10),
            sigma=0.25,
            lower=log_tta_prod_lower,
            upper=log_tta_prod_upper,
        )
        log_tta_leaky_prod = pm.TruncatedNormal(
            "log_tta_leaky_prod",
            mu=np.log(0.15),
            sigma=0.3,
            lower=log_tta_leaky_lower,
            upper=log_tta_leaky_upper,
        )
        log_tta_deg = pm.TruncatedNormal(
            "log_tta_deg",
            mu=np.log(0.05),
            sigma=0.05,
            lower=log_tta_deg_lower,
            upper=log_tta_deg_upper,
        )
        log_tta_kd = pm.TruncatedNormal(
            "log_tta_kd",
            mu=np.log(5),
            sigma=0.25,
            lower=log_kd_lower,
            upper=log_kd_upper,
        )
        log_dox_kd = pm.TruncatedNormal(
            "log_dox_kd",
            mu=np.log(5),
            sigma=0.25,
            lower=log_kd_lower,
            upper=log_kd_upper,
        )
        log_cno_ec50 = pm.TruncatedNormal(
            "log_cno_ec50",
            mu=np.log(5),
            sigma=0.25,
            lower=log_kd_lower,
            upper=log_kd_upper,
        )
        log_clz_ec50 = pm.TruncatedNormal(
            "log_clz_ec50",
            mu=np.log(5),
            sigma=0.25,
            lower=log_kd_lower,
            upper=log_kd_upper,
        )
        log_dreadd_prod = pm.TruncatedNormal(
            "log_dreadd_prod",
            mu=np.log(10),
            sigma=0.25,
            lower=log_dreadd_prod_lower,
            upper=log_dreadd_prod_upper,
        )
        log_dreadd_ec50 = pm.TruncatedNormal(
            "log_dreadd_ec50",
            mu=np.log(10),
            sigma=0.25,
            lower=log_kd_lower,
            upper=log_kd_upper,
        )

        var_obs = pm.HalfNormal("sigma_obs", sigma=0.3)

        mean_plasma_rma = pm.Deterministic(
            "mu",
            mechanistic_expectation(
                log_prod_mouse,
                log_leaky_prod_mouse,
                log_bbb,
                log_deg,
                log_tta_prod,
                log_tta_leaky_prod,
                log_tta_deg,
                log_tta_kd,
                log_dox_kd,
                log_cno_ec50,
                log_clz_ec50,
                log_dreadd_prod,
                log_dreadd_ec50,
            ),
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
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            init="adapt_diag",
            random_seed=42,
            return_inferencedata=True,
            target_accept=0.95,
        )

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y"],
            random_seed=42,
            return_inferencedata=True,
        )
    return brain_dox_ss, cno_pk_model, dox_pk_model, idata, plasma_dox_ss, ppc


@app.cell
def _(az, idata):
    summary = az.summary(
        idata,
        var_names=[
            "mu_log_prod",
            "mu_leaky_log_prod",
            "log_bbb",
            "log_deg",
            "log_tta_prod",
            "log_tta_leaky_prod",
            "log_tta_deg",
            "log_tta_kd",
            "log_dox_kd",
            "log_cno_ec50",
            "log_clz_ec50",
            "log_dreadd_prod",
            "log_dreadd_ec50",
        ],
        round_to=4,
    )
    summary
    return (summary,)


@app.cell
def _(data_dir, summary):
    summary.to_csv(data_dir + "/parameter_summary.csv")
    return


@app.cell
def _(az, data_dir, idata, os, plt):
    az.plot_trace(
        idata,
        var_names=[
            "mu_log_prod",
            "mu_leaky_log_prod",
            "log_bbb",
            "log_deg",
            "log_tta_prod",
            "log_tta_leaky_prod",
            "log_tta_deg",
            "log_tta_kd",
            "log_dox_kd",
            "log_cno_ec50",
            "log_clz_ec50",
            "log_dreadd_prod",
            "log_dreadd_ec50",
        ],
    )

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "trace_plots.svg"))
    plt.gcf()
    return


@app.cell
def _(
    Kvaerno3,
    Model,
    State,
    az,
    brain_dox_ss,
    cno_pk_model,
    dox_pk_model,
    idata,
    n_mice,
    np,
    plasma_dox_ss,
):
    log_prod_samples = idata.posterior["log_prod_mouse"].values
    log_leaky_prod_samples = idata.posterior["log_leaky_prod_mouse"].values
    log_bbb_samples = idata.posterior["log_bbb"].values
    log_deg_samples = idata.posterior["log_deg"].values
    log_tta_prod_samples = idata.posterior["log_tta_prod"].values
    log_tta_leaky_prod_samples = idata.posterior["log_tta_leaky_prod"].values
    log_tta_deg_samples = idata.posterior["log_tta_deg"].values
    log_tta_kd_samples = idata.posterior["log_tta_kd"].values
    log_dox_kd_samples = idata.posterior["log_dox_kd"].values
    log_cno_ec50_samples = idata.posterior["log_cno_ec50"].values
    log_clz_ec50_samples = idata.posterior["log_clz_ec50"].values
    log_dreadd_prod_samples = idata.posterior["log_dreadd_prod"].values
    log_dreadd_ec50_samples = idata.posterior["log_dreadd_ec50"].values

    def plasma_rma_fit(
        prod_samples,
        leaky_prod_samples,
        bbb_samples,
        deg_samples,
        tta_prod_samples,
        tta_leaky_prod_samples,
        tta_deg_samples,
        tta_kd_samples,
        dox_kd_samples,
        cno_ec50_samples,
        clz_ec50_samples,
        dreadd_prod_samples,
        dreadd_ec50_samples,
    ):
        log_prod = prod_samples.reshape(-1, n_mice)
        log_leaky_prod = leaky_prod_samples.reshape(-1, n_mice)
        log_bbb = bbb_samples.reshape(-1)
        log_deg = deg_samples.reshape(-1)
        log_tta_prod = tta_prod_samples.reshape(-1)
        log_tta_leaky_prod = tta_leaky_prod_samples.reshape(-1)
        log_tta_deg = tta_deg_samples.reshape(-1)
        log_tta_kd = tta_kd_samples.reshape(-1)
        log_dox_kd = dox_kd_samples.reshape(-1)
        log_cno_ec50 = cno_ec50_samples.reshape(-1)
        log_clz_ec50 = clz_ec50_samples.reshape(-1)
        log_dreadd_prod = dreadd_prod_samples.reshape(-1)
        log_dreadd_ec50 = dreadd_ec50_samples.reshape(-1)

        total_draws = log_prod.shape[0]

        # Collect all trajectories per species: (total_draws * n_mice, n_times)
        plasma_rma_traj = []
        tta_traj = []
        brain_dox_traj = []
        brain_cno_traj = []
        brain_clz_traj = []

        for i in range(total_draws):
            bbb = np.exp(log_bbb[i])
            deg = np.exp(log_deg[i])
            tta_prod = np.exp(log_tta_prod[i])
            tta_leaky_prod = np.exp(log_tta_leaky_prod[i])
            tta_deg = np.exp(log_tta_deg[i])
            tta_kd = np.exp(log_tta_kd[i])
            dox_kd = np.exp(log_dox_kd[i])
            cno_ec50 = np.exp(log_cno_ec50[i])
            clz_ec50 = np.exp(log_clz_ec50[i])
            dreadd_prod = np.exp(log_dreadd_prod[i])
            dreadd_ec50 = np.exp(log_dreadd_ec50[i])

            state = State(
                brain_dox=brain_dox_ss, plasma_dox=plasma_dox_ss, dreadd=dreadd_prod
            )

            for mouse in range(n_mice):
                prod = np.exp(log_prod[i, mouse])
                leaky_prod = np.exp(log_leaky_prod[i, mouse])

                model = Model(
                    rma_prod=prod,
                    leaky_rma_prod=leaky_prod,
                    rma_bbb_transport=bbb,
                    rma_deg=deg,
                    tta_prod=tta_prod,
                    leaky_tta_prod=tta_leaky_prod,
                    tta_deg=tta_deg,
                    tta_kd=tta_kd,
                    dox_pk_model=dox_pk_model,
                    dox_tta_kd=dox_kd,
                    cno_pk_model=cno_pk_model,
                    cno_ec50=cno_ec50,
                    clz_ec50=clz_ec50,
                    dreadd_prod=dreadd_prod,
                    dreadd_ec50=dreadd_ec50,
                )

                solution = model.solve(0, 96, 1, state, Kvaerno3())
                plasma_rma_traj.append(solution.plasma_rma)
                tta_traj.append(solution.tta)
                brain_dox_traj.append(solution.brain_dox)
                brain_cno_traj.append(solution.brain_cno)
                brain_clz_traj.append(solution.brain_clz)

        # Convert to arrays for statistics
        plasma_rma_arr = np.array(plasma_rma_traj)
        tta_arr = np.array(tta_traj)
        brain_dox_arr = np.array(brain_dox_traj)
        brain_cno_arr = np.array(brain_cno_traj)
        brain_clz_arr = np.array(brain_clz_traj)

        # Compute mean and HDI for each species separately
        results = {
            "plasma_rma": {
                "mean": plasma_rma_arr.mean(axis=0),
                "hdi": az.hdi(plasma_rma_arr, hdi_prob=0.94),
            },
            "tta": {
                "mean": tta_arr.mean(axis=0),
                "hdi": az.hdi(tta_arr, hdi_prob=0.94),
            },
            "brain_dox": {
                "mean": brain_dox_arr.mean(axis=0),
                "hdi": az.hdi(brain_dox_arr, hdi_prob=0.94),
            },
            "brain_cno": {
                "mean": brain_cno_arr.mean(axis=0),
                "hdi": az.hdi(brain_cno_arr, hdi_prob=0.94),
            },
            "brain_clz": {
                "mean": brain_clz_arr.mean(axis=0),
                "hdi": az.hdi(brain_clz_arr, hdi_prob=0.94),
            },
        }

        return results

    return (
        log_bbb_samples,
        log_clz_ec50_samples,
        log_cno_ec50_samples,
        log_deg_samples,
        log_dox_kd_samples,
        log_dreadd_ec50_samples,
        log_dreadd_prod_samples,
        log_leaky_prod_samples,
        log_prod_samples,
        log_tta_deg_samples,
        log_tta_kd_samples,
        log_tta_leaky_prod_samples,
        log_tta_prod_samples,
        plasma_rma_fit,
    )


@app.cell
def _(az, fit_df, mouse_id, n_mice, np, obs_plasma_rma, obs_time, plt, ppc):
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
    # plt.savefig(data_dir + "/per_mouse_posterior_mean.svg")
    plt.gcf()
    return


@app.cell
def _(
    log_bbb_samples,
    log_clz_ec50_samples,
    log_cno_ec50_samples,
    log_deg_samples,
    log_dox_kd_samples,
    log_dreadd_ec50_samples,
    log_dreadd_prod_samples,
    log_leaky_prod_samples,
    log_prod_samples,
    log_tta_deg_samples,
    log_tta_kd_samples,
    log_tta_leaky_prod_samples,
    log_tta_prod_samples,
    plasma_rma_fit,
):
    results = plasma_rma_fit(
        log_prod_samples,
        log_leaky_prod_samples,
        log_bbb_samples,
        log_deg_samples,
        log_tta_prod_samples,
        log_tta_leaky_prod_samples,
        log_tta_deg_samples,
        log_tta_kd_samples,
        log_dox_kd_samples,
        log_cno_ec50_samples,
        log_clz_ec50_samples,
        log_dreadd_prod_samples,
        log_dreadd_ec50_samples,
    )

    pop_plasma_rma = results["plasma_rma"]["mean"]
    pop_plasma_rma_hdi = results["plasma_rma"]["hdi"]
    pop_tta = results["tta"]["mean"]
    pop_tta_hdi = results["tta"]["hdi"]
    pop_brain_dox = results["brain_dox"]["mean"]
    pop_brain_dox_hdi = results["brain_dox"]["hdi"]
    pop_brain_cno = results["brain_cno"]["mean"]
    pop_brain_cno_hdi = results["brain_cno"]["hdi"]
    pop_brain_clz = results["brain_clz"]["mean"]
    pop_brain_clz_hdi = results["brain_clz"]["hdi"]
    return (
        pop_brain_clz,
        pop_brain_clz_hdi,
        pop_brain_cno,
        pop_brain_cno_hdi,
        pop_brain_dox,
        pop_brain_dox_hdi,
        pop_plasma_rma,
        pop_plasma_rma_hdi,
        pop_tta,
        pop_tta_hdi,
    )


@app.cell
def _(pop_brain_clz, pop_brain_cno, pop_brain_dox, pop_plasma_rma, pop_tta):
    print("plasma_rma:", pop_plasma_rma.shape)
    print("tta:", pop_tta.shape)
    print("brain_dox:", pop_brain_dox.shape)
    print("brain_cno:", pop_brain_cno.shape)
    print("brain_clz:", pop_brain_clz.shape)
    return


@app.cell
def _():
    return


@app.cell
def _(plt, pop_plasma_rma):
    plt.plot(pop_plasma_rma, label="plasma_rma")
    # plt.plot(pop_tta, label="tta")
    # plt.plot(pop_brain_dox, label="brain_dox")
    # plt.plot(pop_brain_cno, label="brain_cno")
    # plt.plot(pop_brain_clz, label="brain_clz")
    plt.legend()
    return


@app.cell
def _(
    data_dir,
    np,
    pop_brain_clz,
    pop_brain_clz_hdi,
    pop_brain_cno,
    pop_brain_cno_hdi,
    pop_brain_dox,
    pop_brain_dox_hdi,
    pop_plasma_rma,
    pop_plasma_rma_hdi,
    pop_tta,
    pop_tta_hdi,
):
    np.save(data_dir + "/pop_plasma_rma.npy", pop_plasma_rma)
    np.save(data_dir + "/pop_plasma_rma_hdi.npy", pop_plasma_rma_hdi)
    np.save(data_dir + "/pop_tta.npy", pop_tta)
    np.save(data_dir + "/pop_tta_hdi.npy", pop_tta_hdi)
    np.save(data_dir + "/pop_brain_dox.npy", pop_brain_dox)
    np.save(data_dir + "/pop_brain_dox_hdi.npy", pop_brain_dox_hdi)
    np.save(data_dir + "/pop_brain_cno.npy", pop_brain_cno)
    np.save(data_dir + "/pop_brain_cno_hdi.npy", pop_brain_cno_hdi)
    np.save(data_dir + "/pop_brain_clz.npy", pop_brain_clz)
    np.save(data_dir + "/pop_brain_clz_hdi.npy", pop_brain_clz_hdi)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
