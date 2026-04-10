import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
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


@app.cell
def _():
    data_dir = os.path.join("notebooks", "data", "aav_rma_timecourse")
    ca1_dir = os.path.join(data_dir, "CA1")
    sn_dir = os.path.join(data_dir, "SN")
    cp_dir = os.path.join(data_dir, "CP")
    _, ca1_summary_df = get_df("CA1", ca1_dir)
    _, sn_summary_df = get_df("SN", sn_dir)
    _, cp_summary_df = get_df("CP", cp_dir)
    return (
        ca1_dir,
        ca1_summary_df,
        cp_dir,
        cp_summary_df,
        data_dir,
        sn_dir,
        sn_summary_df,
    )


@app.cell
def _(data_dir):
    # load saved simulations
    ca1_mean = np.load(os.path.join(data_dir, "CA1", "predicted_mean.npy"))
    ca1_hdi = np.load(os.path.join(data_dir, "CA1", "hdi.npy"))

    sn_mean = np.load(os.path.join(data_dir, "SN", "predicted_mean.npy"))
    sn_hdi = np.load(os.path.join(data_dir, "SN", "hdi.npy"))

    cp_mean = np.load(os.path.join(data_dir, "CP", "predicted_mean.npy"))
    cp_hdi = np.load(os.path.join(data_dir, "CP", "hdi.npy"))
    return ca1_hdi, ca1_mean, cp_hdi, cp_mean, sn_hdi, sn_mean


@app.cell
def _(
    ca1_hdi,
    ca1_mean,
    ca1_summary_df,
    cp_hdi,
    cp_mean,
    cp_summary_df,
    data_dir,
    sn_hdi,
    sn_mean,
    sn_summary_df,
):
    # plot predictions
    colors = sb.color_palette("colorblind", 3)
    shapes = ['o', '^', 's']

    time = np.linspace(0, 505, 505)

    for (mean, hdi, df, label), color, shape in zip([
        (ca1_mean, ca1_hdi, ca1_summary_df, "Hippocampus"),
        (sn_mean,  sn_hdi,  sn_summary_df,  "Midbrain"),
        (cp_mean,  cp_hdi,  cp_summary_df,  "Striatum"),
    ], colors, shapes):
        plt.plot(time, mean, color=color, label=label)
        plt.fill_between(time, hdi[:, 0], hdi[:, 1], color=color, alpha=0.25)
        plt.errorbar(df["time"], df["mean"], yerr=df["std"],
                     fmt=shape, color=color, capsize=3)

    plt.xlabel("Time (hr)")
    plt.ylabel("Concentration (nM)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "constitutive_fit.svg"))
    plt.gca()
    return


@app.function
# calculate R2s

def score(predicted, observed) -> float:
    predicted_rma = [predicted[0], predicted[336], predicted[504]]
    return r2_score(observed, predicted_rma)


@app.cell
def _(
    ca1_mean,
    ca1_summary_df,
    cp_mean,
    cp_summary_df,
    sn_mean,
    sn_summary_df,
):
    ca1_score = score(ca1_mean, ca1_summary_df["mean"])
    sn_score = score(sn_mean, sn_summary_df["mean"])
    cp_score = score(cp_mean, cp_summary_df["mean"])
    average_score = np.mean([ca1_score, sn_score, cp_score])

    print(f"CA1 R2: {ca1_score}")
    print(f"SN R2: {sn_score}")
    print(f"CP R2: {cp_score}")
    print(f"Average R2: {average_score}")
    return


@app.function
def get_parameter(df: pl.DataFrame, name: str, correction: float = 0) -> float:
    log_params = df.filter(pl.col("") == name).select(pl.col("mean"), pl.col("hdi_3%"), pl.col("hdi_97%"))
    params = log_params.map_columns(["mean", "hdi_3%", "hdi_97%"], lambda v: np.exp(v + correction**2/2))
    return params


@app.cell
def _(ca1_param_summary_full):
    ca1_param_summary_full
    return


@app.cell
def _(ca1_dir):
    ca1_param_summary_full = pl.read_csv(os.path.join(ca1_dir, "parameter_fit_summary.csv"))
    ca1_prod = get_parameter(ca1_param_summary_full, "mu_log_prod", correction=0.5)
    ca1_prod
    return (ca1_param_summary_full,)


@app.cell
def _(ca1_param_summary_full):
    ca1_bbb = get_parameter(ca1_param_summary_full, "log_bbb")
    ca1_bbb
    return


@app.cell
def _(ca1_param_summary_full):
    ca1_deg = get_parameter(ca1_param_summary_full, "log_deg")
    ca1_deg
    return


@app.cell
def _(sn_dir):
    sn_param_summary_full = pl.read_csv(os.path.join(sn_dir, "parameter_fit_summary.csv"))
    sn_prod = get_parameter(sn_param_summary_full, "mu_log_prod", correction=0.5)
    sn_prod
    return (sn_param_summary_full,)


@app.cell
def _(sn_param_summary_full):
    sn_bbb = get_parameter(sn_param_summary_full, "log_bbb")
    sn_bbb
    return


@app.cell
def _(sn_param_summary_full):
    sn_deg = get_parameter(sn_param_summary_full, "log_deg")
    sn_deg
    return


@app.cell
def _(cp_dir):
    cp_param_summary_full = pl.read_csv(os.path.join(cp_dir, "parameter_fit_summary.csv"))
    cp_prod = get_parameter(cp_param_summary_full, "mu_log_prod", correction=0.5)
    cp_prod
    return (cp_param_summary_full,)


@app.cell
def _(cp_param_summary_full):
    cp_bbb = get_parameter(cp_param_summary_full, "log_bbb")
    cp_bbb
    return


@app.cell
def _(cp_param_summary_full):
    cp_deg = get_parameter(cp_param_summary_full, "log_deg")
    cp_deg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
