use pyo3::prelude::*;
mod fluc;
use fluc::mc_sigma;
use fluc::mc_sigma_parallel;
use fluc::hc2_parallel;

// Make the module available to Python
#[pymodule]
fn fluc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mc_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(mc_sigma_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(hc2_parallel, m)?)?;
    Ok(())
}
