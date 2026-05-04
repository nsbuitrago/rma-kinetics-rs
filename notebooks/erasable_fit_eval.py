import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sb
    import marimo as mo
    import os
    from sklearn.metrics import r2_score

    from rma_kinetics.models.constitutive import Model, State
    from rma_kinetics.solvers import Solver, Dopri5
    from constitutive_fit import get_df

    sb.set_theme(context="talk", style="ticks", font="Arial")
    plt.rc("axes.spines", top=False, right=False)
    return get_df, np, os, plt, r2_score, sb


@app.cell
def _(get_df, os):
    data_dir = os.path.join("notebooks", "data", "constitutive_ferma")
    hip_dir = os.path.join(data_dir, "hippocampus")
    mid_dir = os.path.join(data_dir, "midbrain")
    str_dir = os.path.join(data_dir, "striatum")
    _, hip_summary_df = get_df("Hippocampus", hip_dir)
    _, mid_summary_df = get_df("Midbrain", mid_dir)
    _, str_summary_df = get_df("Striatum", str_dir)
    return data_dir, hip_summary_df, mid_summary_df, str_summary_df


@app.cell
def _(data_dir, np, os):
    hip_mean = np.load(os.path.join(data_dir, "hippocampus", "predicted_mean.npy"))
    hip_hdi = np.load(os.path.join(data_dir, "hippocampus", "hdi.npy"))

    mid_mean = np.load(os.path.join(data_dir, "midbrain", "predicted_mean.npy"))
    mid_hdi = np.load(os.path.join(data_dir, "midbrain", "hdi.npy"))

    str_mean = np.load(os.path.join(data_dir, "striatum", "predicted_mean.npy"))
    str_hdi = np.load(os.path.join(data_dir, "striatum", "hdi.npy"))

    return hip_hdi, hip_mean, mid_hdi, mid_mean, str_hdi, str_mean


@app.cell
def _(
    data_dir,
    hip_hdi,
    hip_mean,
    hip_summary_df,
    mid_hdi,
    mid_mean,
    mid_summary_df,
    np,
    os,
    plt,
    sb,
    str_hdi,
    str_mean,
    str_summary_df,
):
    # plot predictions
    colors = sb.color_palette("colorblind", 3)
    shapes = ['o', '^', 's']

    time = np.linspace(0, 504.5, 1010)

    for (mean, hdi, df, label), color, shape in zip([
        (hip_mean, hip_hdi, hip_summary_df, "Hippocampus"),
        (mid_mean,  mid_hdi,  mid_summary_df,  "Midbrain"),
        (str_mean,  str_hdi,  str_summary_df,  "Striatum"),
    ], colors, shapes):
        plt.plot(time, mean, color=color, label=label)
        plt.fill_between(time, hdi[:, 0], hdi[:, 1], color=color, alpha=0.25)
        plt.errorbar(df["time"], df["mean"], yerr=df["std"],
                     fmt=shape, color=color, capsize=3)

    plt.xlabel("Time (hr)")
    plt.ylabel("Concentration (nM)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "fit.svg"))
    plt.gca()
    return


@app.cell
def _(r2_score):
    def score(predicted, observed, two_week_pt: bool = False) -> float:
        predicted_rma = [
            predicted[0],
            predicted[336],
            predicted[337],
        ]

        if two_week_pt:
            predicted_rma.append(predicted[672])

        predicted_rma.append(predicted[1008])
        predicted_rma.append(predicted[1009])

        return r2_score(observed, predicted_rma)

    return (score,)


@app.cell
def _(
    hip_mean,
    hip_summary_df,
    mid_mean,
    mid_summary_df,
    np,
    score,
    str_mean,
    str_summary_df,
):
    hip_score = score(hip_mean, hip_summary_df["mean"])
    mid_score = score(mid_mean, mid_summary_df["mean"])
    str_score = score(str_mean, str_summary_df["mean"], two_week_pt=True)

    avg_score = np.mean([hip_score, mid_score, str_score])

    print(f"Hippocampus R2: {hip_score}")
    print(f"Midbrain R2: {mid_score}")
    print(f"Striatum R2: {str_score}")
    print(f"Average R2: {avg_score}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
