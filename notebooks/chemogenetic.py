import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.chemogenetic import Model, State
    from rma_kinetics.models.dox import Model as DoxModel, AccessPeriod
    from rma_kinetics.models.cno import Model as CnoModel, Dose
    from rma_kinetics.solvers import Kvaerno3

    import matplotlib.pyplot as plt
    return AccessPeriod, CnoModel, Dose, DoxModel, Kvaerno3, Model, State, plt


@app.cell
def _(AccessPeriod, CnoModel, Dose, DoxModel, Kvaerno3, Model, State):
    # setup dox model
    dox_schedule = AccessPeriod(40, 0, 24)
    dox_pk = DoxModel(schedule=[dox_schedule])

    # setup cno model
    cno_dose = Dose(0.03, 48)
    cno_pk = CnoModel(doses=[cno_dose])

    # setup chemogenetic model
    model = Model(dox_pk_model=dox_pk, cno_pk_model=cno_pk)
    # initial state
    init_state = State()
    solver = Kvaerno3()
    solution = model.solve(0, 96, 1, init_state, solver)
    return (solution,)


@app.cell
def _(plt, solution):
    plt.subplot(2, 2, 1)
    plt.plot(solution.ts, solution.plasma_rma, 'k')
    plt.ylabel('Plasma RMA (nM)')
    plt.xlabel('Time (hr)')

    plt.subplot(2, 2, 2)
    plt.plot(solution.ts, solution.tta, 'k')
    plt.ylabel('tTA (nM)')
    plt.xlabel('Time (hr)')

    plt.subplot(2, 2, 3)
    plt.plot(solution.ts, solution.brain_dox, 'k')
    plt.ylabel('Brain Dox (nM)')
    plt.xlabel('Time (hr)')

    plt.subplot(2, 2, 4)
    plt.plot(solution.ts, solution.brain_clz, 'k')
    plt.ylabel('Brain CLZ (nM)')
    plt.xlabel('Time (hr)')

    plt.tight_layout()
    plt.gcf()

    return


if __name__ == "__main__":
    app.run()
