import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sb
    import marimo as mo
    import os
    from sklearn.metrics import r2_score

    from rma_kinetics.models.chemogenetic import Model, State
    from rma_kinetics.models.dox import Model as DoxModel, AccessPeriod
    from rma_kinetics.models.cno import Model as CnoModel, Dose
    from rma_kinetics.solvers import Kvaerno3
    from utils import rlu_to_nm

    sb.set_theme(context="talk", style="ticks")
    plt.rc("axes.spines", top=False, right=False)
    return np, os, pl, plt, r2_score, rlu_to_nm, sb


@app.cell
def _(os):
    data_dir = os.path.join("notebooks", "data", "chemogenetic")
    return (data_dir,)


@app.cell
def _(os, pl, rlu_to_nm):
    def get_df(data_dir: str, cno_dose: float) -> (pl.DataFrame, pl.DataFrame):
        raw_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
        df = rlu_to_nm(raw_df)
        df_filtered = df.filter(pl.col("cno_dose") == cno_dose)

        summary_df = (
            df_filtered.group_by("time")
            .agg(
                [
                    pl.col("concentration").mean().alias("mean"),
                    pl.col("concentration").std().alias("std"),
                ]
            )
            .sort("time")
        )

        return df_filtered, summary_df

    return (get_df,)


@app.cell
def _(data_dir, get_df, plt, sb):
    df, summary_df = get_df(data_dir, 1)
    df_plot = df.unpivot(
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
    return df, summary_df


@app.cell
def _(data_dir, np, os):
    plasma_rma_mean = np.load(os.path.join(data_dir, "pop_plasma_rma.npy"))
    plasma_rma_hdi = np.load(os.path.join(data_dir, "pop_plasma_rma_hdi.npy"))
    time = np.linspace(0, 97, 97)
    return plasma_rma_hdi, plasma_rma_mean, time


@app.cell
def _(df, os, plasma_rma_hdi, plasma_rma_mean, plt, summary_df, time):
    #plt.vlines(48, ymin=-5, ymax=40, color='lightgrey', linestyle=':')
    plt.plot(time[:-1], plasma_rma_mean[:-1], 'k')
    plt.fill_between(
        time[:-1], plasma_rma_hdi[:-1, 0], plasma_rma_hdi[:-1, 1],
        color='k', alpha=0.25
    )

    # times in the df are post CNO injection, so we need to add 24 hrs to it to compare
    # to the full simulation
    plt.scatter([t + 48 for t in df["time"]], df["concentration"], marker='o', alpha=0.25, color='#E89A2A')
    plt.errorbar([t + 48 for t in summary_df["time"]], summary_df["mean"], yerr=summary_df["std"], fmt='s', color='#E89A2A')
    plt.tight_layout()
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    plt.ylim([-2, 45])
    plt.tight_layout()
    plt.savefig(os.path.join("notebooks", "data", "chemogenetic", "plasma_rma_trajectory.svg"))
    plt.gcf()
    return


@app.cell
def _(np, pl, r2_score):
    def score(df: pl.DataFrame, predictions: np.typing.NDArray) -> float:
        y_true = df["mean"].to_numpy()
        return r2_score(y_true, predictions)

    return (score,)


@app.cell
def _(plasma_rma_mean, score, summary_df):
    predictions = [plasma_rma_mean[48], plasma_rma_mean[72], plasma_rma_mean[96]]
    r2 = score(summary_df, predictions)
    print(f"R^2: {r2:.3f}")
    return


@app.cell
def _(data_dir, np, os):
    brain_dox_mean = np.load(os.path.join(data_dir, "pop_brain_dox.npy"))
    braind_dox_hdi = np.load(os.path.join(data_dir, "pop_brain_dox_hdi.npy"))
    tta_mean = np.load(os.path.join(data_dir, "pop_tta.npy"))
    tta_hdi = np.load(os.path.join(data_dir, "pop_tta_hdi.npy"))
    brain_clz = np.load(os.path.join(data_dir, "pop_brain_clz.npy"))
    brain_clz_hdi = np.load(os.path.join(data_dir, "pop_brain_clz_hdi.npy"))
    return (
        brain_clz,
        brain_clz_hdi,
        brain_dox_mean,
        braind_dox_hdi,
        tta_hdi,
        tta_mean,
    )


@app.cell
def _(brain_clz, np, time, tta_mean):
    # calculate max and Tmax for CLZ and tTA
    clz_tmax = time[np.argmax(brain_clz)]
    tta_tmax = time[np.argmax(tta_mean)]

    clz_max = np.max(brain_clz)
    tta_max = np.max(tta_mean)


    print(f"CLZ max: {clz_max:.3f} at {clz_tmax:.3f} hr")
    print(f"tTA max: {tta_max:.3f} at {tta_tmax:.2f} hr")
    return


@app.cell
def _(brain_dox_mean, braind_dox_hdi, plt, time):
    plt.plot(time[:-1], brain_dox_mean[:-1], 'k--')
    plt.fill_between(
        time[:-1], braind_dox_hdi[:-1, 0], braind_dox_hdi[:-1, 1],
        color='k', alpha=0.25
    )
    return


@app.cell
def _(brain_clz, brain_clz_hdi, os, plt, time, tta_hdi, tta_mean):
    _fig, _ax = plt.subplots(1, 2, figsize=(6.4, 3.5))
    _ax[0].plot(time, brain_clz, 'k')
    _ax[0].fill_between(
        time, brain_clz_hdi[:, 0], brain_clz_hdi[:, 1],
        color='k', alpha=0.25
    )

    _ax[0].set_xlabel("Time (hr)")
    _ax[0].set_ylabel("Brain CLZ (nM)")

    _ax[1].plot(time, tta_mean, 'k')
    _ax[1].fill_between(
        time, tta_hdi[:, 0], tta_hdi[:, 1],
        color='k', alpha=0.25
    )
    _ax[1].set_xlabel("Time (hr)")
    _ax[1].set_ylabel("tTA (nM)")

    plt.tight_layout()
    plt.savefig(os.path.join("notebooks", "data", "chemogenetic", "cno_tta_trajectory.svg"))
    plt.gcf()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
