import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.cno import Model, State, Dose, create_cno_schedule
    from rma_kinetics.solvers import Dopri5

    import matplotlib.pyplot as plt
    import marimo as mo
    return Dopri5, Dose, Model, State, create_cno_schedule, plt


@app.cell
def _(Dopri5, Dose, Model, State):
    dose = Dose(0.03, 1.25)
    cno_model = Model([dose])
    init_state = State()

    solver = Dopri5()
    solution = cno_model.solve(0, 24, 0.1, init_state, solver) 
    return init_state, solution, solver


@app.cell
def _(plt, solution):
    plt.plot(solution.ts, solution.brain_clz, 'k')
    plt.xlabel('Time (hr)')
    plt.ylabel('Brain CLZ (nM)')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(Model, create_cno_schedule, init_state, solver):
    schedule = create_cno_schedule(0.05, 1, 1, 12)
    repeated_dose_model = Model(schedule)
    repeated_solution = repeated_dose_model.solve(0, 48, 0.1, init_state, solver)
    return (repeated_solution,)


@app.cell
def _(plt, repeated_solution):
    plt.plot(repeated_solution.ts, repeated_solution.brain_clz, 'k')
    plt.xlabel('Time (hr)')
    plt.ylabel('Brain CLZ (nM)')
    plt.tight_layout()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
