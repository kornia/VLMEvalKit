use pyo3::prelude::*;

mod smolvlm;
mod smolvlm2;

/// A Python module implemented in Rust.
#[pymodule]
fn kornia_vlm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smolvlm::generate, m)?)?;
    m.add_function(wrap_pyfunction!(smolvlm::generate_raw, m)?)?;
    m.add_class::<smolvlm::SmolVLMInterface>()?;
    m.add_class::<smolvlm2::SmolVLM2Interface>()?;
    Ok(())
}
