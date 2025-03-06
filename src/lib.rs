use pyo3::prelude::*;
mod fluc;
use fluc::mc_sigma;
use fluc::mc_sigma_parallel;

// Make the module available to Python
#[pymodule]
fn _fluctuoscopy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mc_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(mc_sigma_parallel, m)?)?;
    Ok(())
}
