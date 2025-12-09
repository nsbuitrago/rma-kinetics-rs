import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.oscillation import Model, State
    from rma_kinetics.solvers import Dopri5

    import matplotlib.pyplot as plt
    return Dopri5, Model, State, plt


@app.cell
def _(Dopri5, Model, State):
    model = Model(freq=1/24)
    init_state = State()

    t0 = 0; tf = 168; dt = 1;
    solution = model.solve(t0, tf, dt, init_state, solver=Dopri5())
    determinstic = solution.plasma_rma
    noise_strength = 0.5
    solution.apply_noise(noise_strength)
    return determinstic, noise_strength, solution


@app.cell
def _(determinstic, noise_strength, plt, solution):
    plt.plot(solution.ts, determinstic, 'k', label="Deterministic")
    plt.plot(
        solution.ts,
        solution.plasma_rma,
        'lightgrey',
        label=f"sigma = {noise_strength}"
    )

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA")
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
