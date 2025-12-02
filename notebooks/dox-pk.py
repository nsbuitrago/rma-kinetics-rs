import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import dox
    from rma_kinetics.solvers import Dopri5

    import marimo as mo
    import matplotlib.pyplot as plt
    return Dopri5, dox, plt


@app.cell
def _(Dopri5, dox, plt):
    dox_access = dox.AccessPeriod(40, 0, 48) # 40mg/kg food administered over 48 hours
    dox_pk_model = dox.Model(schedule=[dox_access])
    init_state = dox.State()

    solver = Dopri5()
    solution = dox_pk_model.solve(0, 96, 1, init_state, solver)
    plt.plot(solution.ts, solution.brain_dox, 'k')
    plt.xlabel("Time (hr)")
    plt.ylabel("Dox brain concentration (nM)")
    plt.tight_layout()
    plt.gcf()
    return init_state, solver


@app.cell
def _(dox, init_state, plt, solver):
    dose_schedule = dox.create_dox_schedule(40, 0, 24, 1, 24)
    dox_pk_model_repeated = dox.Model(schedule=dose_schedule)
    solution_repeated = dox_pk_model_repeated.solve(0, 96, 1, init_state, solver)
    plt.plot(solution_repeated.ts, solution_repeated.brain_dox, 'k')
    plt.xlabel("Time (hr)")
    plt.ylabel("Dox brain concentration (nM)")
    plt.tight_layout()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
