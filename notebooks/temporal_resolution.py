import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from rma_kinetics.models.oscillation import Model, State
    from rma_kinetics.solvers import Dopri5
    from jax import config as jax_config, numpy as jnp, random
    from jax.scipy.signal import welch
    from SALib.sample import latin
    import numpy as np
    from scipy import signal as sp_signal
    import seaborn as sb
    import matplotlib.pyplot as plt
    import polars as pl
    import os

    data_dir = os.path.join("notebooks", "data", "temporal_resolution")
    jax_config.update("jax_enable_x64", True)
    sb.set_theme("talk", "ticks")
    sb.set_palette("crest_r")
    return (
        Dopri5,
        Model,
        State,
        data_dir,
        jnp,
        latin,
        mo,
        np,
        os,
        pl,
        plt,
        random,
        sb,
        sp_signal,
        welch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RMA Kinetics Analysis
    """)
    return


@app.cell
def _(Model, State, jnp, plt):
    def plot_deg_rate_sweep(
        half_lives: list[float], rma_params: list[float], sim_params: dict[str, float]
    ):
        for half_life in half_lives:
            deg_rate = jnp.log(2) / half_life
            sim_params["init_state"] = State(
                rma_params[0] / rma_params[2], rma_params[0] / deg_rate
            )
            model = Model(rma_params[0], rma_params[1], rma_params[2], deg_rate)
            solution = model.solve(**sim_params)

            plt.plot(
                solution.ts,
                solution.plasma_rma / (rma_params[0] / deg_rate),
                label=f"{half_life}",
            )

    return (plot_deg_rate_sweep,)


@app.cell
def _(Dopri5, data_dir, os, plot_deg_rate_sweep, plt, sb):
    max_rma_prod_rate = 0.2  # nM/hr
    rma_rt_rate = 0.6  # 1/hr
    rma_half_lives = [100, 50, 25, 12.5]  # hrs
    oscillation_freq = 1 / 72  # 1/hr

    sweep_sim_config = {"t0": 0, "tf": 504, "dt": 1, "solver": Dopri5()}

    plot_deg_rate_sweep(
        rma_half_lives,
        [max_rma_prod_rate, oscillation_freq, rma_rt_rate],
        sweep_sim_config,
    )
    sb.despine()
    plt.xlabel("Time (hr)")
    plt.ylabel("Normalized Plasma RMA (A.U.)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "normalized_rma_varying_deg_rate.svg"))
    plt.gca()
    return (
        max_rma_prod_rate,
        oscillation_freq,
        rma_half_lives,
        rma_rt_rate,
        sweep_sim_config,
    )


@app.cell
def _(Model, State, jnp, plt):
    def plot_freq_sweep(
        freqs: list[float],
        half_lives: list[float],
        rma_params: list[float],
        sim_params: dict[str, float],
    ):
        markers = ["s", "^", "d", "o"]
        for half_life, marker in zip(half_lives, markers):
            dyn_range = []
            for freq in freqs:
                deg_rate = jnp.log(2) / half_life
                sim_params["init_state"] = State(
                    rma_params[0] / rma_params[1], rma_params[0] / deg_rate
                )
                model = Model(rma_params[0], freq, rma_params[1], deg_rate)
                solution = model.solve(**sim_params)
                norm_plasma_rma = solution.plasma_rma / (rma_params[0] / deg_rate)

                dyn_range.append(jnp.max(norm_plasma_rma) - jnp.min(norm_plasma_rma))

            plt.plot(freqs, dyn_range, label=f"{half_life}", marker=marker)

    return (plot_freq_sweep,)


@app.cell
def _(
    data_dir,
    max_rma_prod_rate,
    os,
    plot_freq_sweep,
    plt,
    rma_half_lives,
    rma_rt_rate,
    sb,
    sweep_sim_config,
):
    freqs = [1 / 72, 1 / 48, 1 / 24, 1 / 12, 1 / 6]
    plot_freq_sweep(
        freqs, rma_half_lives, [max_rma_prod_rate, rma_rt_rate], sweep_sim_config
    )
    sb.despine()
    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Dynamic Range (A.U.)")
    plt.legend(frameon=False, title="RMA Half-Life")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "dynamic_range_varying_deg_and_freq.svg"))
    plt.gca()
    return


@app.cell
def _(Model, State, jnp, plt):
    def plot_prod_rate_sweep(
        rma_prod_rates: list[float],
        half_lives: list[float],
        rma_params: list[float],
        sim_params: dict[str, float],
    ):
        markers = ["s", "^", "d", "o"]
        for half_life, marker in zip(half_lives, markers):
            max_rma = []
            deg_rate = jnp.log(2) / half_life
            for prod_rate in rma_prod_rates:
                sim_params["init_state"] = State(
                    prod_rate / rma_params[0], prod_rate / deg_rate
                )
                model = Model(prod_rate, rma_params[0], deg_rate, rma_params[1])
                solution = model.solve(**sim_params)

                max_rma.append(jnp.max(solution.plasma_rma))

            plt.plot(rma_prod_rates, max_rma, label=f"{half_life}", marker=marker)

    return (plot_prod_rate_sweep,)


@app.cell
def _(
    data_dir,
    os,
    oscillation_freq,
    plot_prod_rate_sweep,
    plt,
    rma_half_lives,
    rma_rt_rate,
    sb,
    sweep_sim_config,
):
    plot_prod_rate_sweep(
        [7e-3, 1e-2, 4e-2, 7e-2],
        rma_half_lives,
        [oscillation_freq, rma_rt_rate],
        sweep_sim_config,
    )

    sb.despine()
    plt.xlabel("RMA Production Rate (nM/hr)")
    plt.ylabel("Max Plasma RMA (nM)")
    plt.legend(frameon=False, title="RMA Half-Life")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "max_intensity_varying_deg_and_prod.svg"))
    plt.gca()
    return


@app.cell
def _(
    Model,
    State,
    jnp,
    max_rma_prod_rate,
    oscillation_freq,
    rma_rt_rate,
    sweep_sim_config,
):
    def max_v_range_map(half_life: float):
        max_rma_conc = []
        dyn_range = []
        deg_rate = jnp.log(2) / half_life
        sweep_sim_config["init_state"] = State(
            max_rma_prod_rate / rma_rt_rate, max_rma_prod_rate / deg_rate
        )
        model = Model(max_rma_prod_rate, oscillation_freq, rma_rt_rate, deg_rate)
        solution = model.solve(**sweep_sim_config)

        max_rma = jnp.max(solution.plasma_rma)
        norm_plasma_rma = solution.plasma_rma / (max_rma_prod_rate / deg_rate)
        dyn_range = jnp.max(norm_plasma_rma) - jnp.min(norm_plasma_rma)

        return max_rma, dyn_range

    return (max_v_range_map,)


@app.cell
def _(data_dir, jnp, latin, max_v_range_map, os, plt, sb):
    half_life_space = {
        "num_vars": 1,
        "names": ["rma_half_life"],
        "bounds": [[12.5, 100]],
    }

    half_life_vector = latin.sample(half_life_space, 1000)

    results = [max_v_range_map(half_life[0]) for half_life in half_life_vector]
    max_rma = jnp.array([result[0] for result in results])
    dyn_range = jnp.array([result[1] for result in results])
    plt.scatter(
        dyn_range,
        max_rma,
        marker="o",
        s=12,
        c=[p[0] for p in half_life_vector],
        cmap="crest",
    )
    cbar = plt.colorbar()
    cbar.set_label("RMA Half-Life (hr)", rotation=270, labelpad=25)

    plt.ylabel("Max Concentration (nM)")
    plt.xlabel("Dynamic Range (A.U.)")
    sb.despine()
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, "max_intensity_v_dyn_range_varying_half_life.svg")
    )
    plt.gca()
    return


@app.cell
def _(Model, State, jnp, plt):
    def plot_noisy_rma_example(
        rma_params: list[float],
        noise_std: float,
        sim_config: dict[str, float],
        prng_key,
    ):
        sim_config["init_state"] = State(0, 0)
        deg_rate = jnp.log(2) / rma_params[-2]
        rma_params[-2] = deg_rate
        model = Model(*rma_params)
        solution = model.solve(**sim_config)
        deterministic = solution
        # solution.apply_noise(noise_std)

        # plt.plot(solution.ts, solution.plasma_rma, color='lightgrey', label=rf"$\sigma = {noise_std}$")
        plt.plot(
            solution.ts, deterministic.plasma_rma, label="Deterministic", color="black"
        )

    return


@app.cell
def _(
    Model,
    State,
    data_dir,
    jnp,
    os,
    plt,
    rma_half_lives,
    rma_rt_rate,
    sb,
    sweep_sim_config,
):
    noisy_rma_example_oscillation_freq = 1 / 100
    sweep_sim_config["init_state"] = State(0, 0)
    deg_rate = jnp.log(2) / rma_half_lives[0]
    example_model = Model(
        7e-3, noisy_rma_example_oscillation_freq, rma_rt_rate, deg_rate
    )
    solution = example_model.solve(**sweep_sim_config)
    noise_std = 0.1
    deterministic = solution.plasma_rma
    solution.apply_noise(noise_std)
    noisy = jnp.clip(solution.plasma_rma, a_min=0, a_max=None)
    plt.plot(solution.ts, noisy, label=f"$\sigma$ = {noise_std}", color="lightgrey")
    plt.plot(solution.ts, deterministic, label="Deterministic", color="black")
    # plot_noisy_rma_example([max_rma_prod_rate, noisy_rma_example_oscillation_freq, rma_rt_rate, rma_half_lives[0]], noise_std=0.2, sim_config=sweep_sim_config, prng_key=random.key(111))

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "example_noisy_plasma_rma.svg"))
    plt.gca()
    return


@app.cell
def _(jnp, random):
    def apply_noise(solution, std, prng_key):
        """
        Apply Gaussian noise to a given trajectory.

        Args:
            solution (jax.Array): Solution/trajectory to apply noise to.
            std (float): Noise standard deviation.
            prng_key (jax.Array): Jax PRNG key.

        Returns:
            noisy_solution (jax.Array): Solution with applied Gaussian noise.
        """
        noise = std * random.normal(prng_key, shape=(len(solution),))
        return jnp.clip(solution * (1 + noise), min=0)

    return (apply_noise,)


@app.cell
def _(
    Dopri5,
    Model,
    State,
    apply_noise,
    jnp,
    max_rma_prod_rate,
    plt,
    random,
    rma_half_lives,
    rma_rt_rate,
    welch,
):
    init_state_f_recovery = State(
        max_rma_prod_rate / rma_rt_rate,
        max_rma_prod_rate / (jnp.log(2) / rma_half_lives[0]),
    )

    def freq_recovery_single_iter():
        deg_rate = jnp.log(2) / rma_half_lives[0]
        target_freq = 1 / 15

        model = Model(max_rma_prod_rate, target_freq, rma_rt_rate, deg_rate)

        n_cycles = 50
        fs = 10
        t0 = 0
        tf = n_cycles // target_freq

        solution = model.solve(
            t0, tf, dt=1 / fs, init_state=init_state_f_recovery, solver=Dopri5()
        )

        norm_plasma_rma = solution.plasma_rma / (max_rma_prod_rate / deg_rate)
        noisy_rma = apply_noise(norm_plasma_rma, std=0.1, prng_key=random.key(123))
        nperseg = len(noisy_rma) // 2
        f, psd = welch(noisy_rma - jnp.mean(noisy_rma), fs=fs, nperseg=nperseg)
        _, psd_clean = welch(norm_plasma_rma, fs=fs, nperseg=nperseg)
        peak_idx = jnp.argmax(psd)
        fpeak = f[peak_idx]
        psd_peak = psd[peak_idx]
        freq_match = jnp.isclose(fpeak, target_freq, rtol=0.05, atol=0)
        psd_noise = jnp.where(~jnp.isclose(f, target_freq, rtol=0.05), psd, jnp.nan)
        snr = psd_peak / jnp.nanmean(psd_noise)
        print(fpeak)
        print(freq_match)
        print(snr)

        plt.plot(f, psd, "lightgrey", label="Noise, $\sigma = 0.2$")
        plt.plot(f, psd_clean, "k", label="Deterministic", alpha=0.5)
        # plt.plot(solution.ts, norm_plasma_rma)
        # plt.plot(solution.ts, noisy_rma, color='lightgrey')
    return freq_recovery_single_iter, init_state_f_recovery


@app.cell
def _(data_dir, freq_recovery_single_iter, os, plt, sb):

    freq_recovery_single_iter()
    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Power Spectral Density (PSD)")
    plt.tight_layout()
    sb.despine()
    plt.xlim(0, 0.5)
    # plot vertical line at oscillation frequency
    plt.savefig(os.path.join(data_dir, "example_noisy_psd_for_resolution.svg"))
    plt.gca()
    return


@app.cell
def _(apply_noise, jnp, welch):
    def freq_recovery_inner(
        model,
        simulation_config,
        noise_std,
        n_iter,
        prng_keys,
        rtol,
        fs,
        min_snr,
        rma_steady_state,
        target_freq,
    ):
        resolution = 0
        for i in range(0, n_iter):
            solution = model.solve(**simulation_config)
            norm_plasma_rma = solution.plasma_rma / rma_steady_state
            norm_plasma_rma = apply_noise(norm_plasma_rma, noise_std, prng_keys[i])
            nperseg = len(norm_plasma_rma) // 2

            freq, psd = welch(
                norm_plasma_rma - jnp.mean(norm_plasma_rma), fs=fs, nperseg=nperseg
            )
            peak_idx = jnp.argmax(psd)
            fpeak = freq[peak_idx]
            psd_peak = psd[peak_idx]
            freq_match = jnp.isclose(fpeak, target_freq, rtol=rtol, atol=0)
            psd_noise = jnp.where(
                ~jnp.isclose(freq, target_freq, rtol=rtol), psd, jnp.nan
            )
            snr = psd_peak / jnp.nanmean(psd_noise)

            if freq_match and snr >= min_snr:
                resolution += 1
        return resolution / n_iter

    return (freq_recovery_inner,)


@app.cell
def _(
    Dopri5,
    Model,
    State,
    data_dir,
    freq_recovery_inner,
    init_state_f_recovery,
    jnp,
    max_rma_prod_rate,
    os,
    pl,
    plt,
    random,
    rma_half_lives,
    rma_rt_rate,
    sb,
):
    # iterate over frequency range and fixed noise std
    def freq_recovery():
        # deg_rate = jnp.log(2) / rma_half_lives[0] # 100 hr serum half-life
        target_freqs = jnp.linspace(1 / 30, 1 / 3, 10)
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "dt": 1 / fs,
            "init_state": init_state_f_recovery,
            "solver": Dopri5(),
        }

        markers = ["s", "^", "d", "o"]
        for i, half_life in enumerate(rma_half_lives):
            percent_freq_recovery = []
            deg_rate = jnp.log(2) / half_life
            simulation_config["init_state"] = State(
                max_rma_prod_rate / rma_rt_rate, max_rma_prod_rate / deg_rate
            )
            for target_freq in target_freqs:
                prng_keys = random.split(
                    random.key(i * 10_000 + int(float(target_freq) * 1_000_000)),
                    num=500,
                )
                model = Model(max_rma_prod_rate, target_freq, rma_rt_rate, deg_rate)
                simulation_config["tf"] = n_cycles // float(target_freq)
                recovery = freq_recovery_inner(
                    model,
                    simulation_config,
                    noise_std=0.05,  # fixed noise
                    n_iter=500,
                    prng_keys=prng_keys,
                    rtol=0.05,
                    fs=fs,
                    min_snr=2,
                    rma_steady_state=max_rma_prod_rate / deg_rate,
                    target_freq=target_freq,
                )

                percent_freq_recovery.append(recovery * 100)

            recovery_df = pl.DataFrame(
                {
                    "Frequency": list(target_freqs),
                    "Percent Recovery": percent_freq_recovery,
                }
            )

            recovery_df.write_parquet(
                os.path.join(
                    data_dir, f"202604_freq_recovery_deg_rate_{deg_rate}.parquet"
                )
            )
            plt.plot(target_freqs, percent_freq_recovery, marker=markers[i])

            plt.xlabel("Frequency (1/hr)")
            plt.ylabel("Frequency Recovery (%)")
            # plt.legend(["100", "50", "25", "12.5"], title="RMA Half-Life (hr)", frameon=False)
            plt.tight_layout()
            sb.despine()
            plt.savefig(os.path.join(data_dir, "202604_freq_recovery_all.svg"))
            plt.gca()

    freq_recovery()
    return


@app.cell
def _(jnp):
    def bandpower(power, f, frange, include=True):
        in_band = jnp.logical_and(f >= frange[0], f <= frange[1])
        mask = in_band if include else ~in_band
        if jnp.sum(mask) < 2:
            return jnp.nan
        return jnp.trapezoid(power[mask], x=f[mask])

    def forcing_signal(ts, prod, freq):
        return prod * (1 + jnp.sin(2 * jnp.pi * freq * ts))

    return bandpower, forcing_signal


@app.cell
def _(apply_noise, bandpower, jnp, np, sp_signal):
    def power_cutoff_inner(
        model,
        simulation_config,
        noise_std,
        n_iter,
        prng_keys,
        target_freq,
        fs,
        rtol,
        rma_steady_state,
    ):
        ratios = []
        frange = (target_freq * (1 - rtol), target_freq * (1 + rtol))
        for i in range(n_iter):
            solution = model.solve(**simulation_config)
            norm_plasma_rma = solution.plasma_rma / rma_steady_state
            noisy_rma = apply_noise(norm_plasma_rma, noise_std, prng_keys[i])
            nperseg = len(noisy_rma) // 2
            freq, psd = sp_signal.welch(
                np.asarray(noisy_rma - jnp.mean(noisy_rma)),
                fs=fs,
                nperseg=nperseg,
            )
            freq = jnp.array(freq)
            psd = jnp.array(psd)
            avg_band_power = bandpower(psd, freq, frange, include=True)
            total_band_power = jnp.sum(psd) * (freq[1] - freq[0])
            ratio = avg_band_power / jnp.maximum(total_band_power, 1e-12)
            ratios.append(ratio)

        ratios = jnp.array(ratios)
        finite = ratios[jnp.isfinite(ratios)]
        if len(finite) == 0:
            return jnp.nan
        return jnp.mean(finite)

    return (power_cutoff_inner,)


@app.cell
def _(apply_noise, forcing_signal, jnp, np, sp_signal):
    def coherence_cutoff_inner(
        model,
        simulation_config,
        noise_std,
        n_iter,
        prng_keys,
        target_freq,
        fs,
        rtol,
        prod,
        deg_rate,
    ):
        # deterministic forcing from model parameters, sampled on the same grid as output
        ts = np.arange(
            simulation_config["t0"],
            simulation_config["tf"] + simulation_config["dt"],
            simulation_config["dt"],
        )
        x = forcing_signal(jnp.array(ts), prod, target_freq)
        x = np.asarray(x)
        nperseg = len(x) // 2
        fx, input_psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)

        coh_scores = []
        for i in range(n_iter):
            solution = model.solve(**simulation_config)
            y = apply_noise(solution.plasma_rma, noise_std, prng_keys[i])
            y = np.asarray(y / (prod / deg_rate))

            fy, pyy = sp_signal.welch(y, fs=fs, nperseg=nperseg)
            fxy, pxy = sp_signal.csd(x, y, fs=fs, nperseg=nperseg)

            if not (np.array_equal(fx, fy) and np.array_equal(fx, fxy)):
                continue

            coh = (np.abs(pxy) ** 2) / (input_psd * pyy + 1e-12)
            target_idx = np.argmin(np.abs(fx - target_freq))
            coh_scores.append(coh[target_idx])

        if len(coh_scores) == 0:
            return jnp.nan
        return jnp.array(np.nanmean(coh_scores))

    return (coherence_cutoff_inner,)


@app.cell
def _(
    Dopri5,
    Model,
    State,
    data_dir,
    jnp,
    max_rma_prod_rate,
    os,
    pl,
    power_cutoff_inner,
    random,
    rma_half_lives,
    rma_rt_rate,
):
    def power_cutoff():
        target_freqs = jnp.linspace(1 / 30, 1 / 3, 10)
        noise_stds = jnp.linspace(0, 0.2, 20)
        deg_rate = jnp.log(2) / rma_half_lives[0]
        n_cycles = 50
        fs = 10

        simulation_config = {
            "t0": 0,
            "dt": 1 / fs,
            "init_state": State(
                max_rma_prod_rate / rma_rt_rate, max_rma_prod_rate / deg_rate
            ),
            "solver": Dopri5(),
        }

        avg_power = []
        freqs = []
        noise = []
        for fi, f in enumerate(target_freqs):
            simulation_config["tf"] = n_cycles // float(f)
            model = Model(max_rma_prod_rate, float(f), rma_rt_rate, deg_rate)
            for ni, n in enumerate(noise_stds):
                n_iter = 100 if float(n) > 0 else 1
                prng_keys = random.split(
                    random.key(fi * 10_000 + ni * 1_000 + 7), num=n_iter
                )
                power = power_cutoff_inner(
                    model,
                    simulation_config,
                    noise_std=float(n),
                    n_iter=n_iter,
                    prng_keys=prng_keys,
                    target_freq=float(f),
                    fs=fs,
                    rtol=0.05,
                    rma_steady_state=max_rma_prod_rate / deg_rate,
                )
                avg_power.append(power)
                freqs.append(float(f))
                noise.append(float(n))

        power_df = pl.DataFrame(
            {
                "Frequency": freqs,
                "Noise": noise,
                "Average Power": avg_power,
            }
        )
        power_df.write_parquet(
            os.path.join(data_dir, f"202604_power_cutoff_deg_rate_{deg_rate}.parquet")
        )

        return power_df

    power_df = power_cutoff()
    return (power_df,)


@app.cell
def _(data_dir, jnp, os, pl, plt, power_df, sb):
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    def power_ratio():
        target_freqs = jnp.linspace(1 / 30, 1 / 3, 10)
        cmap = cm.get_cmap("flare_r")
        norm = Normalize(vmin=target_freqs.min(), vmax=target_freqs.max())

        power_df_sub = None
        for f in target_freqs:
            power_df_sub = power_df.filter(pl.col("Frequency") == float(f))
            color = cmap(norm(float(f)))
            plt.plot(power_df_sub["Noise"], power_df_sub["Average Power"], color=color)

        plt.xlabel("Noise Standard Deviation")
        plt.ylabel("Relative Power")
        sb.despine()
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Frequency (1/hr)")
        cbar.set_label("Frequency (1/hr)", rotation=270, labelpad=25)
        if power_df_sub is not None:
            plt.plot(
                power_df_sub["Noise"],
                power_df_sub["Noise"] / 2,
                "--",
                color="lightgrey",
            )
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "202604_power_ratio_100hr_half_life.svg"))
        plt.gca()

    power_ratio()
    return Normalize, cm


@app.cell
def _(
    Dopri5,
    Model,
    State,
    coherence_cutoff_inner,
    data_dir,
    jnp,
    max_rma_prod_rate,
    os,
    pl,
    random,
    rma_half_lives,
    rma_rt_rate,
):
    def coh_cutoff():
        target_freqs = jnp.linspace(1 / 30, 1 / 3, 10)
        noise_stds = jnp.linspace(0, 0.2, 20)
        deg_rate = jnp.log(2) / rma_half_lives[0]
        n_cycles = 50
        fs = 10

        simulation_config = {
            "t0": 0,
            "dt": 1 / fs,
            "init_state": State(
                max_rma_prod_rate / rma_rt_rate, max_rma_prod_rate / deg_rate
            ),
            "solver": Dopri5(),
        }

        avg_coh = []
        freqs = []
        noise = []
        for fi, f in enumerate(target_freqs):
            simulation_config["tf"] = n_cycles // float(f)
            model = Model(max_rma_prod_rate, float(f), rma_rt_rate, deg_rate)
            for ni, n in enumerate(noise_stds):
                n_iter = 1000 if float(n) > 0 else 1
                prng_keys = random.split(
                    random.key(fi * 10_000 + ni * 1_000 + 29), num=n_iter
                )
                coh = coherence_cutoff_inner(
                    model,
                    simulation_config,
                    noise_std=float(n),
                    n_iter=n_iter,
                    prng_keys=prng_keys,
                    target_freq=float(f),
                    fs=fs,
                    rtol=0.25,
                    prod=max_rma_prod_rate,
                    deg_rate=deg_rate,
                )
                avg_coh.append(coh)
                freqs.append(float(f))
                noise.append(float(n))

        coh_df = pl.DataFrame(
            {"Frequency": freqs, "Noise": noise, "Coherence": avg_coh}
        )
        coh_df.write_parquet(
            os.path.join(data_dir, f"202604_coh_cutoff_deg_rate_{deg_rate}.parquet")
        )

        return coh_df

    coh_df = coh_cutoff()
    return (coh_df,)


@app.cell
def _(Normalize, cm, coh_df, data_dir, jnp, os, pl, plt, sb):
    def plot_coherence():
        target_freqs = jnp.linspace(1 / 30, 1 / 3, 10)
        cmap = cm.get_cmap("flare_r")
        norm = Normalize(vmin=target_freqs.min(), vmax=target_freqs.max())

        for f in target_freqs:
            coh_df_sub = coh_df.filter(pl.col("Frequency") == float(f))
            color = cmap(norm(float(f)))
            plt.plot(coh_df_sub["Noise"], coh_df_sub["Coherence"], color=color)

        plt.xlabel("Noise Standard Deviation")
        plt.ylabel("Coherence (R)")
        sb.despine()
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Frequency (1/hr)")
        cbar.set_label("Frequency (1/hr)", rotation=270, labelpad=25)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "202604_coh_100hr_half_life.svg"))
        plt.gca()

    plot_coherence()
    return


@app.cell
def _(coh_df, data_dir, os, pl, plt):
    def plot_cutoff():
        # coh_df = pl.read_parquet(os.path.join(data_dir, "202604_coh"))
        cutoff_coh = (
            coh_df.filter(pl.col("Coherence") > 0.5)
            .group_by("Noise")
            .agg(pl.col("Frequency").max().alias("Frequency"))
            .sort("Noise")
        )

        plt.scatter(cutoff_coh["Noise"], cutoff_coh["Frequency"])
        plt.xlabel("Noise Standard Deviation")
        plt.ylabel("Cutoff Frequency (1/hr)")
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "202604_cutoff_freq_from_coh.svg"))
        plt.gca()

    plot_cutoff()
    return


@app.cell
def _(Model, State, random):
    from SALib.sample import morris as morris_sampler
    from SALib.analyze import morris as morris_analyzer

    def map_temporal_precision(
        params,
        freq_recovery_inner,
        max_rma_prod_rate,
        Dopri5,
    ):
        prod_rate, rt_rate, deg_rate, noise_level = params
        target_freq = 1 / 10
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "tf": n_cycles // target_freq,
            "dt": 1 / fs,
            "init_state": State(prod_rate / rt_rate, prod_rate / deg_rate),
            "solver": Dopri5(),
        }
        model = Model(prod_rate, target_freq, rt_rate, deg_rate)
        prng_keys = random.split(
            random.key(int(prod_rate * 1e8) % (2**31 - 1)), num=100
        )
        recovery = freq_recovery_inner(
            model,
            simulation_config,
            noise_std=noise_level,
            n_iter=100,
            prng_keys=prng_keys,
            rtol=0.05,
            fs=fs,
            min_snr=2,
            rma_steady_state=prod_rate / deg_rate,
            target_freq=target_freq,
        )
        return recovery

    parameter_space = {
        "num_vars": 4,
        "names": ["$k_{RMA}$", "$k_{RT}$", "$\\gamma_{RMA}$", "$\\sigma$"],
        "bounds": [[3.5e-3, 1.05e-2], [0.3, 0.9], [7e-3, 5.5e-2], [0.01, 0.2]],
    }
    return (
        map_temporal_precision,
        morris_analyzer,
        morris_sampler,
        parameter_space,
    )


@app.cell
def _(
    Dopri5,
    freq_recovery_inner,
    jnp,
    map_temporal_precision,
    morris_analyzer,
    morris_sampler,
    parameter_space,
):
    tp_param_vectors = morris_sampler.sample(parameter_space, 250)
    tp_y = jnp.array(
        [
            map_temporal_precision(
                p, freq_recovery_inner, max_rma_prod_rate=7e-3, Dopri5=Dopri5
            )
            for p in tp_param_vectors
        ]
    )
    tp_sens = morris_analyzer.analyze(parameter_space, tp_param_vectors, tp_y)
    return (tp_sens,)


@app.cell
def _(data_dir, jnp, os, parameter_space, plt, sb, tp_sens):
    max_mu_star_tp = max(tp_sens["mu_star"])
    norm_mu_star_tp = tp_sens["mu_star"] / max_mu_star_tp
    norm_mu_star_conf = tp_sens["mu_star_conf"] / max_mu_star_tp
    norm_sigma = tp_sens["sigma"] / jnp.max(tp_sens["sigma"])

    plt.bar(
        parameter_space["names"],
        norm_mu_star_tp,
        yerr=norm_mu_star_conf,
        color="lightgrey",
    )
    plt.ylabel("Relative Ranking")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "202604_norm_morris_mean_tp.svg"))
    plt.gca()

    plt.figure()
    plt.bar(parameter_space["names"], norm_sigma, color="lightgrey")
    plt.ylabel("Relative Nonlinearity or Interaction")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "202604_norm_morris_std_tp.svg"))
    plt.gca()
    return


@app.cell
def _(
    Dopri5,
    Model,
    State,
    coherence_cutoff_inner,
    power_cutoff_inner,
    random,
):
    def map_temporal_resolution(params):
        prod_rate, rt_rate, deg_rate, noise_level = params
        target_freq = 1 / 10
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "tf": n_cycles // target_freq,
            "dt": 1 / fs,
            "init_state": State(prod_rate / rt_rate, prod_rate / deg_rate),
            "solver": Dopri5(),
        }
        model = Model(prod_rate, target_freq, rt_rate, deg_rate)
        prng_keys = random.split(random.key(int(rt_rate * 1e7) % (2**31 - 1)), num=100)
        return power_cutoff_inner(
            model,
            simulation_config,
            noise_std=noise_level,
            n_iter=100,
            prng_keys=prng_keys,
            target_freq=target_freq,
            fs=fs,
            rtol=0.05,
            rma_steady_state=prod_rate / deg_rate,
        )

    def map_coherence(params):
        prod_rate, rt_rate, deg_rate, noise_level = params
        target_freq = 1 / 10
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "tf": n_cycles // target_freq,
            "dt": 1 / fs,
            "init_state": State(prod_rate / rt_rate, prod_rate / deg_rate),
            "solver": Dopri5(),
        }
        model = Model(prod_rate, target_freq, rt_rate, deg_rate)
        prng_keys = random.split(random.key(int(deg_rate * 1e8) % (2**31 - 1)), num=100)
        return coherence_cutoff_inner(
            model,
            simulation_config,
            noise_std=noise_level,
            n_iter=100,
            prng_keys=prng_keys,
            target_freq=target_freq,
            fs=fs,
            rtol=0.25,
            prod=prod_rate,
            deg_rate=deg_rate,
        )

    return map_coherence, map_temporal_resolution


@app.cell
def _(
    data_dir,
    jnp,
    map_coherence,
    map_temporal_resolution,
    morris_analyzer,
    morris_sampler,
    os,
    parameter_space,
    plt,
    sb,
):
    tr_param_vectors = morris_sampler.sample(parameter_space, 250)
    tr_y = jnp.array([map_temporal_resolution(p) for p in tr_param_vectors])
    tr_sens = morris_analyzer.analyze(parameter_space, tr_param_vectors, tr_y)

    max_mu_star_tr = max(tr_sens["mu_star"])
    norm_mu_star_tr = tr_sens["mu_star"] / max_mu_star_tr
    norm_mu_star_conf_tr = tr_sens["mu_star_conf"] / max_mu_star_tr
    norm_sigma_tr = tr_sens["sigma"] / jnp.max(tr_sens["sigma"])

    plt.figure()
    plt.bar(
        parameter_space["names"],
        norm_mu_star_tr,
        yerr=norm_mu_star_conf_tr,
        color="lightgrey",
    )
    plt.ylabel("Relative Ranking")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "202604_norm_morris_mean_tr.svg"))

    plt.figure()
    plt.bar(parameter_space["names"], norm_sigma_tr, color="lightgrey")
    plt.ylabel("Relative Nonlinearity or Interaction")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "202604_norm_morris_std_tr.svg"))

    coh_param_vectors = morris_sampler.sample(parameter_space, 250)
    coh_y = jnp.array([map_coherence(p) for p in coh_param_vectors])
    coh_sens = morris_analyzer.analyze(parameter_space, coh_param_vectors, coh_y)

    max_mu_star_coh = max(coh_sens["mu_star"])
    norm_mu_star_coh = coh_sens["mu_star"] / max_mu_star_coh
    norm_mu_star_conf_coh = coh_sens["mu_star_conf"] / max_mu_star_coh
    norm_sigma_coh = coh_sens["sigma"] / jnp.max(coh_sens["sigma"])

    plt.figure()
    plt.bar(
        parameter_space["names"],
        norm_mu_star_coh,
        yerr=norm_mu_star_conf_coh,
        color="lightgrey",
    )
    plt.ylabel("Relative Ranking")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "202604_norm_morris_mean_coh.svg"))

    plt.figure()
    plt.bar(parameter_space["names"], norm_sigma_coh, color="lightgrey")
    plt.ylabel("Relative Nonlinearity or Interaction")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "202604_norm_morris_std_coh.svg"))
    return


if __name__ == "__main__":
    app.run()
