import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import cno
    from rma_kinetics.solvers import Dopri5

    import matplotlib.pyplot as plt
    import marimo as mo
    return Dopri5, cno, plt


@app.cell
def _(Dopri5, cno):
    dose = cno.Dose(0.03, 1.25)
    cno_model = cno.Model([dose])
    init_state = cno.State()

    solver = Dopri5()
    solution = cno_model.solve(0, 24, 0.1, init_state, solver) 
    return (solution,)


@app.cell
def _(plt, solution):
    plt.plot(solution.ts, solution.brain_clz, 'k')
    plt.xlabel('Time (hr)')
    plt.ylabel('Brain CLZ (nM)')
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(solution):
    len(solution.ts)
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
