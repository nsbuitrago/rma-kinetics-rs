import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    from rma_kinetics.models.constitutive import Model, State
    from rma_kinetics.solvers import Solver, Dopri5
    from utils import rlu_to_nm
    from diffopt.multiswarm import ParticleSwarm, get_best_loss_and_params
    from functools import partial
    from jax import config as jax_config
    from sklearn.metrics import r2_score

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import polars as pl
    import seaborn as sb
    import typer
    import os

    cli = typer.Typer()
    sb.set_theme(context="notebook", style="ticks", font="arial")
    plt.rc("axes.spines", top=False, right=False)
    jax_config.update("jax_enable_x64", True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Constitutive RMA Kinetic Model

    This notebook explores a constitutive model for released markers of activity.
    A simple model is developed and fit to RMA timecourse data measured over the course of three weeks.

    Data for RMA expression in the CA1 region of the hippocampus, caudate putamen, and substantia nigra is available. Select the brain region below to use the corresponding data in the notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    dataset_id = mo.ui.radio(options=["CA1", "CP", "SN"], value="CA1", label="RMA Timecourse Dataset")
    dataset_id
    return (dataset_id,)


@app.cell(hide_code=True)
def _(dataset_id):
    data_dir = os.path.join("notebooks", "data", "aav_rma_timecourse", dataset_id.value)
    return (data_dir,)


@app.function
def get_df(dataset_id: str, data_dir: str):
    raw_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    df = rlu_to_nm(raw_df)
    
    summary_df = df.group_by("time").agg([
        pl.col("concentration").mean().alias("mean"),
        pl.col("concentration").std().alias("std")
    ]).sort("time")

    return df, summary_df


@app.cell(hide_code=True)
def _(data_dir, dataset_id):
    df, summary_df = get_df(dataset_id.value, data_dir)
    df_plot = df.unpivot(
        on=['rlu', 'concentration'],
        index='time',
        variable_name='output_type',
        value_name='value'
    )

    grid = sb.FacetGrid(data=df_plot, col='output_type', sharey=False)
    grid.map(sb.barplot, "time", "value", order=[0, 336, 504], color="lightgrey")

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
    return (summary_df,)


@app.cell
def _():
    mo.md(r"""
    ## Parameter Estimation

    The constitutive model is fit to RMA timecourse data with parameters bound by the known RMA half-life and blood-brain transport kinetics of IgG1 antibodies.
    """)
    return


@app.function
def loss(params, args):
    """
    Compute squared residuals for predicted plasma RMA concentration.

    Arguments
    ---------
    params: tuple[float, float, float]
    args: tuple[np.ndarray, dict[str, any]]

    Returns
    -------
    mse: float
    """
    observed, simulation_config = args
    model = Model(*params)
    solution = model.solve(**simulation_config)
    predicted = [
        solution.plasma_rma[0],
        solution.plasma_rma[2],
        solution.plasma_rma[3],
    ]

    return np.sum((observed - predicted) ** 2)


@app.function
def get_sim_config(
    t0: float = 0,
    t1: float = 504,
    dt: float = 168,
    init_state: State = State(),
    solver: Solver = Dopri5()
):
    return {
        "t0": t0,
        "tf": t1,
        "dt": dt,
        "init_state": init_state,
        "solver": solver
    }


@app.function
@cli.command()
def fit(
    dataset_id: str = "CA1",
    data_dir: str = os.path.join("notebooks", "data", "aav_rma_timecourse")):
    
    sim_config = get_sim_config()
    bounds = [
        (1e-5, 1), # production rate - most flexible since we are unsure about hsyn
        (0.54, 1), # reverse transcytosis based on IgG antibody efflux
        (0.0048630532, 0.0104337257), # degradation based on RMA half-life
    ]

    _, summary_df = get_df(dataset_id, data_dir)
    observed = summary_df.select("mean").to_numpy().squeeze()
    loss_fn = partial(loss, args=(observed, sim_config))
    swarm = ParticleSwarm(
        nparticles=100,
        ndim=3,
        xlow=[b[0] for b in bounds],
        xhigh=[b[1] for b in bounds],
    )
    pso_result = swarm.run_pso(loss_fn)
    best_loss, best_params = get_best_loss_and_params(
        pso_result["swarm_loss_history"],
        pso_result["swarm_x_history"]
    )

    mse = best_loss / 3
    print(f"MSE: {mse:.4f}")
    print(f"Fitted Parameters: {best_params}")

    return mse, best_params


@app.cell
def _(summary_df):
    def inspect_fit(params: np.ndarray, observed: pl.DataFrame):
        sim_config = get_sim_config(dt=1)
        model = Model(*params)
        solution = model.solve(**sim_config) 
    
        predicted = [
            solution.plasma_rma[0],
            solution.plasma_rma[336],
            solution.plasma_rma[504],
        ]

        observed = summary_df.select("mean").to_numpy().squeeze()
        r2 = r2_score(observed, predicted)

        fig = plt.figure(figsize=(6, 4))
        plt.plot(solution.ts, solution.plasma_rma, 'k')
        plt.errorbar(summary_df["time"], summary_df["mean"], yerr=summary_df["std"], fmt="o", color="k")
        plt.xlabel("Time (hr)")
        plt.ylabel("Plasma RMA (nM)")
        plt.title(f"Fitted Constitutive Model (R² = {r2:.3f})")
    
        return fig
    return (inspect_fit,)


@app.cell
def _(data_dir, dataset_id, inspect_fit, summary_df):
    with mo.status.spinner(title="Fitting model") as _spinner:
        mse, fitted_params = fit(dataset_id.value, data_dir)
        _fig = inspect_fit(fitted_params, summary_df)

    plt.gcf()

    return (fitted_params,)


@app.cell
def _(fitted_params):
    mo.md(f"""
    Production Rate (nM/hr): {fitted_params[0]:.6f}

    Reverse Transcytosis Rate (1/hr): {fitted_params[1]:.6f}

    Degradation Rate (1/hr): {fitted_params[2]:.6f}
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
