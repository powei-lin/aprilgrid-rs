use image::imageops::resize;
use image::{GenericImageView, GrayImage, Luma};
use imageproc::filter::median_filter;
use imageproc::morphology::{dilate_mut, erode_mut};

pub mod homography;
pub mod tag_families;

pub fn max_pool(grey_img: &GrayImage, block_size: u32) -> (GrayImage, GrayImage) {
    let h = grey_img.height();
    let w = grey_img.width();
    let (hs, ws) = (h / block_size, w / block_size);

    let mut out_min = GrayImage::new(ws, hs);
    let mut out_max = GrayImage::new(ws, hs);

    for (r_small, r_large) in (0..h).step_by(block_size as usize).enumerate() {
        for (c_small, c_large) in (0..w).step_by(block_size as usize).enumerate() {
            let p = grey_img.view(c_large, r_large, block_size, block_size);
            let mut min_v = 255;
            let mut max_v = 0;
            p.pixels().for_each(|(_, _, v)| {
                let v = v.0[0];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            });
            let coord = (c_small as u32, r_small as u32);
            out_min[coord] = [min_v].into();
            out_max[coord] = [max_v].into();
        }
    }
    (out_min, out_max)
}

pub fn threshold(grey_img: &GrayImage) -> (GrayImage, GrayImage) {
    let tile_size = 4;
    let h = grey_img.height();
    let w = grey_img.width();
    let (hs, hr) = (h / tile_size, h % tile_size);
    let (ws, wr) = (w / tile_size, w % tile_size);

    let (mut min_pool_img, mut max_pool_img) = max_pool(grey_img, tile_size);
    // let max_pool_img = median_filter(&mut max_pool_img, 1, 1);
    // let min_pool_img = median_filter(&mut min_pool_img, 1, 1);
    let new_buf: Vec<u8> = max_pool_img
        .pixels()
        .zip(min_pool_img.pixels())
        .map(|(vmax, vmin)| vmax.0[0] - vmin.0[0])
        .collect();
    let min_white_black_diff = 20;
    let img_diff = GrayImage::from_vec(ws, hs, new_buf).unwrap();
    let mut out_max = GrayImage::new(ws, hs);
    img_diff
        .pixels()
        .zip(out_max.pixels_mut())
        .for_each(|(p0, p1)| {
            if p0.0[0] < min_white_black_diff {
                p1.0[0] = 0;
            } else {
                p1.0[0] = 255;
            }
        });
    // let out_max = resize(&out_max, w - wr, h - hr, image::imageops::FilterType::Nearest);
    // threshim = np.where(im_diff < self.min_white_black_diff, np.uint8(0),
    //                         np.where(im > (im_min + im_diff // 2), np.uint8(255), np.uint8(0)))
    // dilate_mut(&mut out_max, imageproc::distance_transform::Norm::L1, 0);
    (img_diff, out_max)
}
