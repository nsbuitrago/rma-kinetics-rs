import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import tetoff, dox
    from rma_kinetics.solvers import Dopri5

    import matplotlib.pyplot as plt
    import marimo as mo
    return Dopri5, dox, mo, plt, tetoff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple example of a Tet-Off RMA system

    We'll start with an example where RMA is driven by an inducible promoter and tTA is constitutively expressed.
    We will assume we have an initial period of dox administration from T = 0hr to T = 168 hr to allow dox concentrations
    to reach steady state. As dox is cleared out after T = 168 hr, the RMA concentration will increase as tTA becomes available.
    """)
    return


@app.cell
def _(Dopri5, dox, plt, tetoff):
    # model setup
    dox_admin = dox.AccessPeriod(40, start_time=0, stop_time=168)
    dox_pk_model = dox.Model(schedule=[dox_admin])
    tetoff_model = tetoff.Model(dox_pk_model=dox_pk_model)
    init_state = tetoff.State()

    # simulation parameters
    t0 = 0
    tf = 504
    dt = 1
    solver = Dopri5()

    # solve and plot
    solution = tetoff_model.solve(t0, tf, dt, init_state, solver)
    plt.plot(solution.ts, solution.plasma_rma, 'k')
    plt.xlabel("Time (hr)")
    plt.ylabel("[Plasma RMA] (nM)")
    plt.tight_layout()
    plt.gcf()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
