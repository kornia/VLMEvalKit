use kornia_image::allocator::CpuAllocator;
use kornia_io::{
    functional::read_image_any_rgb8,
    jpeg::{read_image_jpeg_mono8, read_image_jpeg_rgb8},
    png::{read_image_png_mono8, read_image_png_rgb8},
};
use kornia_vlm::smolvlm2::{InputMedia, SmolVlm2, SmolVlm2Config};
use pyo3::prelude::*;
use std::path::Path;

#[pyclass]
pub struct SmolVLM2Interface {
    model: SmolVlm2<CpuAllocator>,
}

/// Smart image reader that automatically detects format and attempts multiple reading strategies
fn read_image_smart(
    file_path: &str,
) -> Result<kornia_image::Image<u8, 3, CpuAllocator>, kornia_io::error::IoError> {
    let path = Path::new(file_path);

    // First try the generic reader which handles format detection automatically
    if let Ok(image) = read_image_any_rgb8(file_path) {
        return Ok(image);
    }

    // If that fails, try specific readers based on file extension
    if let Some(extension) = path.extension() {
        let ext_lower = extension.to_string_lossy().to_lowercase();

        match ext_lower.as_ref() {
            "jpg" | "jpeg" => {
                // Try RGB first, then convert mono to RGB if needed
                if let Ok(image) = read_image_jpeg_rgb8(file_path) {
                    return Ok(image);
                }
                // If RGB fails, try reading as mono and convert to RGB
                if let Ok(mono_image) = read_image_jpeg_mono8(file_path) {
                    // Convert mono (1 channel) to RGB (3 channels) by replicating the channel
                    let (height, width) = (mono_image.rows(), mono_image.cols());
                    let mono_data = mono_image.as_slice();
                    let mut rgb_data = Vec::with_capacity(height * width * 3);

                    for pixel in mono_data {
                        rgb_data.push(*pixel); // R
                        rgb_data.push(*pixel); // G
                        rgb_data.push(*pixel); // B
                    }

                    let rgb_image = kornia_image::Image::<u8, 3, _>::new(
                        kornia_image::ImageSize { height, width },
                        rgb_data,
                        CpuAllocator,
                    )
                    .map_err(kornia_io::error::IoError::ImageCreationError)?;

                    return Ok(rgb_image);
                }
            }
            "png" => {
                // Try RGB first, then convert mono to RGB if needed
                if let Ok(image) = read_image_png_rgb8(file_path) {
                    return Ok(image);
                }
                // If RGB fails, try reading as mono and convert to RGB
                if let Ok(mono_image) = read_image_png_mono8(file_path) {
                    // Convert mono (1 channel) to RGB (3 channels) by replicating the channel
                    let (height, width) = (mono_image.rows(), mono_image.cols());
                    let mono_data = mono_image.as_slice();
                    let mut rgb_data = Vec::with_capacity(height * width * 3);

                    for pixel in mono_data {
                        rgb_data.push(*pixel); // R
                        rgb_data.push(*pixel); // G
                        rgb_data.push(*pixel); // B
                    }

                    let rgb_image = kornia_image::Image::<u8, 3, _>::new(
                        kornia_image::ImageSize { height, width },
                        rgb_data,
                        CpuAllocator,
                    )
                    .map_err(kornia_io::error::IoError::ImageCreationError)?;

                    return Ok(rgb_image);
                }
            }
            _ => {
                // For unknown extensions, try the generic reader one more time
                return read_image_any_rgb8(file_path);
            }
        }
    }

    // If all else fails, return an error
    Err(kornia_io::error::IoError::InvalidFileExtension(
        path.to_path_buf(),
    ))
}

#[pymethods]
impl SmolVLM2Interface {
    #[new]
    fn new() -> Self {
        Self {
            model: SmolVlm2::new(SmolVlm2Config {
                do_sample: false, // set to false for greedy decoding
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

        // read the images using smart reader with format detection and fallbacks
        let images = image_paths
            .into_iter()
            .map(|p| read_image_smart(&p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_krn_error)?;

        // generate a caption of the image
        let caption = self
            .model
            .inference_raw(
                &text_prompt,
                InputMedia::Images(images),
                sample_length,
                CpuAllocator,
            )
            .map_err(map_error)?;

        Ok(caption)
    }

    fn clear_context(&mut self) -> PyResult<String> {
        self.model.clear_context().unwrap();
        Ok("Context cleared".to_string())
    }
}
