import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.oscillation import State, StochasticModel
    from rma_kinetics.solvers import Euler

    import seaborn as sb
    import matplotlib.pyplot as plt
    import os

    sb.set_theme(context="talk", style="ticks", font="Arial")
    plt.rc("axes.spines", top=False, right=False)
    data_dir = os.path.join("notebooks", "data", "temporal_resolution")
    return Euler, State, StochasticModel, data_dir, os, plt


@app.cell
def _(Euler, State, StochasticModel, data_dir, os, plt):
    # sanity check
    _model = StochasticModel(prod_noise=0)
    _clean_solution = _model.solve(0, 500, dt=1/10, init_state=State(), solver=Euler(dt0=0.1))

    _model = StochasticModel(prod_noise=2)

    for i in range(5):
        _noisy_solution = _model.solve(0, 500, dt=1/10, init_state=State(), solver=Euler(dt0=0.1))
        plt.plot(_noisy_solution.ts, _noisy_solution.plasma_rma, alpha=0.5)

    plt.plot(_clean_solution.ts, _clean_solution.plasma_rma, label="Deterministic", color="k", linewidth=3)
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "stochastic_model_example.svg"))
    plt.gcf()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
