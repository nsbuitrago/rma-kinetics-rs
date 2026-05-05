#![cfg(feature = "py")]

use super::{
    CNOModel, DoxModel, SensitivityEngine, DEFAULT_CLZ_COOPERATIVITY, DEFAULT_CNO_COOPERATIVITY,
    DEFAULT_DREADD_COOPERATIVITY, DEFAULT_TTA_COOPERATIVITY,
};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};

const N_GLOBAL_PARAMS: usize = 11;
const PENALTY_MU: f64 = 1e12;

#[pyclass(name = "AdjointEngine")]
#[derive(Clone)]
pub struct AdjointEngine {
    inner: SensitivityEngine,
}

#[pymethods]
impl AdjointEngine {
    #[new]
    #[pyo3(signature = (mouse_id, obs_time, n_mice, dox_pk_model=DoxModel::default(), cno_pk_model=CNOModel::default(), plasma_dox_ss=0.0, brain_dox_ss=0.0, t0=0.0, dt_sub=0.25, tta_cooperativity=DEFAULT_TTA_COOPERATIVITY, cno_cooperativity=DEFAULT_CNO_COOPERATIVITY, clz_cooperativity=DEFAULT_CLZ_COOPERATIVITY, dreadd_cooperativity=DEFAULT_DREADD_COOPERATIVITY))]
    fn new(
        mouse_id: PyReadonlyArray1<'_, i64>,
        obs_time: PyReadonlyArray1<'_, f64>,
        n_mice: usize,
        dox_pk_model: DoxModel,
        cno_pk_model: CNOModel,
        plasma_dox_ss: f64,
        brain_dox_ss: f64,
        t0: f64,
        dt_sub: f64,
        tta_cooperativity: f64,
        cno_cooperativity: f64,
        clz_cooperativity: f64,
        dreadd_cooperativity: f64,
    ) -> PyResult<Self> {
        let inner = SensitivityEngine::new(
            mouse_id,
            obs_time,
            n_mice,
            dox_pk_model,
            cno_pk_model,
            plasma_dox_ss,
            brain_dox_ss,
            t0,
            dt_sub,
            tta_cooperativity,
            cno_cooperativity,
            clz_cooperativity,
            dreadd_cooperativity,
        )?;

        Ok(Self { inner })
    }

    #[pyo3(signature = (log_prod_mouse, log_leaky_prod_mouse, log_bbb, log_deg, log_tta_prod, log_tta_leaky_prod, log_tta_deg, log_tta_kd, log_dox_kd, log_cno_ec50, log_clz_ec50, log_dreadd_prod, log_dreadd_ec50))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        log_prod_mouse: PyReadonlyArray1<'_, f64>,
        log_leaky_prod_mouse: PyReadonlyArray1<'_, f64>,
        log_bbb: f64,
        log_deg: f64,
        log_tta_prod: f64,
        log_tta_leaky_prod: f64,
        log_tta_deg: f64,
        log_tta_kd: f64,
        log_dox_kd: f64,
        log_cno_ec50: f64,
        log_clz_ec50: f64,
        log_dreadd_prod: f64,
        log_dreadd_ec50: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.predict_with_jacobian(
            py,
            log_prod_mouse,
            log_leaky_prod_mouse,
            log_bbb,
            log_deg,
            log_tta_prod,
            log_tta_leaky_prod,
            log_tta_deg,
            log_tta_kd,
            log_dox_kd,
            log_cno_ec50,
            log_clz_ec50,
            log_dreadd_prod,
            log_dreadd_ec50,
        ) {
            Ok((mu, _, _, _)) => Ok(mu),
            Err(_) => Ok(PyArray1::from_vec(py, vec![PENALTY_MU; self.inner.n_obs])),
        }
    }

    #[pyo3(signature = (g_mu, log_prod_mouse, log_leaky_prod_mouse, log_bbb, log_deg, log_tta_prod, log_tta_leaky_prod, log_tta_deg, log_tta_kd, log_dox_kd, log_cno_ec50, log_clz_ec50, log_dreadd_prod, log_dreadd_ec50))]
    fn vjp<'py>(
        &self,
        py: Python<'py>,
        g_mu: PyReadonlyArray1<'_, f64>,
        log_prod_mouse: PyReadonlyArray1<'_, f64>,
        log_leaky_prod_mouse: PyReadonlyArray1<'_, f64>,
        log_bbb: f64,
        log_deg: f64,
        log_tta_prod: f64,
        log_tta_leaky_prod: f64,
        log_tta_deg: f64,
        log_tta_kd: f64,
        log_dox_kd: f64,
        log_cno_ec50: f64,
        log_clz_ec50: f64,
        log_dreadd_prod: f64,
        log_dreadd_ec50: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let g = g_mu.as_array().iter().copied().collect::<Vec<f64>>();
        let n_mice = self.inner.n_mice;

        match self.inner.predict_with_jacobian(
            py,
            log_prod_mouse,
            log_leaky_prod_mouse,
            log_bbb,
            log_deg,
            log_tta_prod,
            log_tta_leaky_prod,
            log_tta_deg,
            log_tta_kd,
            log_dox_kd,
            log_cno_ec50,
            log_clz_ec50,
            log_dreadd_prod,
            log_dreadd_ec50,
        ) {
            Ok((_, jac_prod, jac_leaky, jac_global)) => {
                let jac_prod = jac_prod.readonly();
                let jac_leaky = jac_leaky.readonly();
                let jac_global = jac_global.readonly();

                let mut d_prod = vec![0.0; n_mice];
                let mut d_leaky = vec![0.0; n_mice];
                let mut d_global = vec![0.0; N_GLOBAL_PARAMS];

                for obs_idx in 0..self.inner.n_obs {
                    let g_obs = g.get(obs_idx).copied().unwrap_or(0.0);
                    for m in 0..n_mice {
                        d_prod[m] += jac_prod.as_array()[[obs_idx, m]] * g_obs;
                        d_leaky[m] += jac_leaky.as_array()[[obs_idx, m]] * g_obs;
                    }

                    for k in 0..N_GLOBAL_PARAMS {
                        d_global[k] += jac_global.as_array()[[obs_idx, k]] * g_obs;
                    }
                }

                Ok((
                    PyArray1::from_vec(py, d_prod),
                    PyArray1::from_vec(py, d_leaky),
                    PyArray1::from_vec(py, d_global),
                ))
            }
            Err(_) => Ok((
                PyArray1::from_vec(py, vec![0.0; n_mice]),
                PyArray1::from_vec(py, vec![0.0; n_mice]),
                PyArray1::from_vec(py, vec![0.0; N_GLOBAL_PARAMS]),
            )),
        }
    }

    #[getter]
    fn get_n_obs(&self) -> usize {
        self.inner.n_obs
    }

    #[getter]
    fn get_n_mice(&self) -> usize {
        self.inner.n_mice
    }
}
