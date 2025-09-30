use kornia_image::allocator::CpuAllocator;
use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_vlm::smolvlm2::{utils::SmolVlm2Config, SmolVlm2};
use pyo3::prelude::*;

#[pyclass]
pub struct SmolVLM2Interface {
    model: SmolVlm2<CpuAllocator>,
}

#[pymethods]
impl SmolVLM2Interface {
    #[new]
    fn new() -> Self {
        Self {
            model: SmolVlm2::new(SmolVlm2Config {
                do_sample: false, // set to false for greedy decoding
                seed: 420,
                ..Default::default()
            })
            .unwrap(),
        }
    }

    #[pyo3(signature = (text_prompt, sample_length, image_paths=vec![]))]
    fn generate_raw(
        &mut self,
        text_prompt: String,
        sample_length: usize,
        image_paths: Vec<String>,
    ) -> PyResult<String> {
        // NOTE: e is inferred and there's two types of error to handle (SmolVLM & kornia io)
        let map_error = |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e));
        let map_krn_error =
            |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e));

        // read the image
        let images = image_paths
            .into_iter()
            .map(|p| read_image_jpeg_rgb8(&p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_krn_error)?;

        // generate a caption of the image
        let caption = self
            .model
            .inference_raw(&text_prompt, images, sample_length, CpuAllocator)
            .map_err(map_error)?;

        Ok(caption)
    }

    fn clear_context(&mut self) -> PyResult<String> {
        self.model.clear_context().unwrap();
        Ok("Context cleared".to_string())
    }
}
