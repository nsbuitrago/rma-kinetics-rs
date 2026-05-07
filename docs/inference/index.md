# Inference

Parameter estimation with PyMC is supported through the `InferenceSolver` classes.
For example, to fit the constitutive model to experimental data using PyMC, we can import
the `InferenceSolver` or `PopulationInferenceSolver` class. These classes can be wrapped
in `InferenceOp` and `PopulationInferenceOp`, respectively so that the models can work
with PyMC and PyTensor.

## Examples

In the following example, we fit the constitutive model with data in the form of a `polars.DataFrame`.

```python
from rma_kinetics.models.constitutive import PopulationInferenceSolver, State
from rma_kinetics.pymc import PopulationInferenceOp
import pymc as pm
import polars as pl
# 
# ... define your data

def fit(data: pl.DataFrame):
    """
    Fit constitutive model to data with PyMC for parameter variability estimation.

    Parameters
    ----------
    data : pl.DataFrame
        The experimental data to fit.
    """
    # create an inference solver and wrap with a PyMC compatible op
    inference_solver = PopulationInferenceSolver(
        mouse_id=data["mouse_id"].to_numpy().astype(int),
        obs_time=data["time"].to_numpy(),
        n_mice=n_mice,
        init_state=State(),
        t0=0.0,
        tf=504,
        dt=1,
    )
    
    predict_op = PopulationInferenceOp(inference_solver)

    # setup PyMC model and parameters
    obs_id = np.arange(n_obs)
    coords = {
        "mouse": np.arange(n_mice),
        "obs_id": obs_id,
    }
    
    with pm.Model(coords=coords):
        mean_log_prod = pm.TruncatedNormal(
            "mu_log_prod",
            mu=np.log(0.2),
            sigma=0.35,
            lower=np.log(1e-4),
            upper=np.log(10),
        )
    
        log_prod_mouse = pm.TruncatedNormal(
            "log_prod_mouse",
            mu=mean_log_prod,
            sigma=0.5,
            lower=np.log(1e-4),
            upper=np.log(10),
            dims="mouse",
        )
    
        # population-level priors (log-normal parameterization)
        log_bbb = pm.TruncatedNormal(
            "log_bbb",
            mu=np.log(0.6),
            sigma=0.15,
            lower=np.log(1e-2),
            upper=np.log(10),
        )
        log_deg = pm.TruncatedNormal(
            "log_deg",
            mu=np.log(0.007),
            sigma=0.36,
            lower=np.log(1e-4),
            upper=np.log(1e-2),
        )
        # compute the predicted plasma RMA for each observation
        mu = predict_op(log_prod_mouse, log_bbb, log_deg)
    
        var_obs = pm.HalfNormal("sigma_obs", sigma=0.3)

        # sample
        pm.Normal("y", mu=mu, sigma=var_obs, observed=obs_plasma_rma, dims="obs_id")
    
        idata = pm.sample(
            draws=100,
            tune=50,
            chains=1,
            cores=1,
            return_inferencedata=True,
        )

```

For a full examples, see any of the [`_fit.py` notebooks](https://github.com/allenai/rma-kinetics-rs/tree/main/notebooks).

## Usage

::: rma_kinetics.pymc.InferenceOp

::: rma_kinetics.pymc.PopulationInferenceOp
