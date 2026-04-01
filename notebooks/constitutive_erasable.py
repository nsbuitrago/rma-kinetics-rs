import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.constitutive import erasable
    from rma_kinetics.solvers import Dopri5

    import seaborn as sb
    import matplotlib.pyplot
    import marimo as mo
    import os

    from constitutive_fit import get_df

    return Dopri5, erasable, get_df, mo, os, sb


@app.cell
def _(erasable):
    tev_schedule = erasable.create_tev_schedule(100, 168, repeat=1, interval=336)
    model = erasable.Model(tev_schedule)
    return (model,)


@app.cell
def _(Dopri5, erasable, model):
    solution = model.solve(0, 504.5, 0.25, erasable.State(), Dopri5())
    return (solution,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(plt, solution):
    plt.plot(solution.ts, solution.plasma_rma, 'k')
    plt.vlines([168, 504], 0, 25, linestyles='--', color="lightgrey")
    plt.xlabel("Time (hr)")
    plt.ylabel("Concentration (nM)")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    dataset_id = mo.ui.radio(options=["hippocampus", "midbrain", "striatum"], value="hippocampus", label="RMA Timecourse Dataset")
    dataset_id
    return (dataset_id,)


@app.cell
def _(dataset_id, os):
    data_dir = os.path.join("notebooks", "data", "constitutive_ferma", dataset_id.value)
    return (data_dir,)


@app.cell
def _(data_dir, dataset_id, get_df, plt, sb):
    df, summary_df = get_df(dataset_id.value, data_dir)
    df_plot = df.unpivot(
        on=["rlu", "concentration"],
        index="time",
        variable_name="output_type",
        value_name="value",
    )

    grid = sb.FacetGrid(data=df_plot, col="output_type", sharey=False)
    grid.map(sb.pointplot, "time", "value", order=[0, 168, 168.5, 504, 504.5])

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
    return


@app.cell
def _():
    # 2mg/ml - 160µL TEV dose
    # ~320 ng
    # 27,000 g/mol * 10^9 = ng/mol / 10^9 = ng/nmol
    # 0.0118518519 nmol
    return


@app.cell
def _():
    # rma prod - 0.3
    # bbb transport - 0.6
    # deg 0.007
    # tev amount is fixed
    # tev degradation - 
    # tev cut rate - 
    return


if __name__ == "__main__":
    app.run()
