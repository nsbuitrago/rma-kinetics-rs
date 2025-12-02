import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import constitutive
    from rma_kinetics.solvers import Dopri5
    import matplotlib.pyplot as plt
    return Dopri5, constitutive, plt


@app.cell
def _(Dopri5, constitutive):
    # run a simple simulation
    model = constitutive.Model()
    init_state = constitutive.State()
    t0 = 0
    t1 = 504
    dt = 1
    solver = Dopri5()

    solution = model.solve(t0, t1, dt, init_state, solver)
    return (solution,)


@app.cell
def _(plt, solution):
    # plot the solution
    plt.plot(solution.ts, solution.plasma_rma)
    plt.plot(solution.ts, solution.brain_rma)
    return


@app.cell
def _(solution):
    solution.ts
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
