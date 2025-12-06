use image::{ImageBuffer, Luma};
use rayon::prelude::*;
use crate::image_util::GrayImagef32;

pub fn gaussian_blur_par(img: &GrayImagef32, sigma: f32) -> GrayImagef32 {
    let width = img.width() as usize;
    let height = img.height() as usize;
    
    // Create kernel
    let radius = (2.0 * sigma).ceil() as isize;
    let kernel_len = (2 * radius + 1) as usize;
    let mut kernel = Vec::with_capacity(kernel_len);
    let sigma2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    
    for i in -radius..=radius {
        let v = (-(i * i) as f32 / sigma2).exp();
        kernel.push(v);
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }

    // Horizontal pass
    // We allocate a temp buffer of f32 directly
    let mut temp = vec![0.0f32; width * height];
    let in_raw = img.as_raw(); // Vec<f32>
    
    // Safety: Pointers
    let in_ptr = in_raw.as_ptr() as usize;
    let temp_slice = temp.as_mut_slice();

    temp_slice
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let in_ptr = in_ptr as *const f32;
            let row_start = y * width;
            
            for (x, out_val) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..kernel_len {
                    let offset = k as isize - radius;
                    let sample_x = (x as isize + offset).clamp(0, width as isize - 1) as usize;
                    unsafe {
                        acc += *in_ptr.add(row_start + sample_x) * kernel[k];
                    }
                }
                *out_val = acc;
            }
        });

    // Vertical pass
    let mut out = GrayImagef32::new(img.width(), img.height());
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr(), width * height) };
    let temp_ptr = temp.as_ptr() as usize;

    out_slice
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let temp_ptr = temp_ptr as *const f32;

            for (x, out_val) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..kernel_len {
                    let offset = k as isize - radius;
                    let sample_y = (y as isize + offset).clamp(0, height as isize - 1) as usize;
                    unsafe {
                        acc += *temp_ptr.add(sample_y * width + x) * kernel[k];
                    }
                }
                *out_val = acc;
            }
        });

    out
}
