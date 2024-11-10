use faer;
use image::{GenericImage, GenericImageView};
pub type GrayImagef32 = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;

pub fn tag_homography(corners: &[(f32, f32)], side_bits: u8, margin: f32) -> faer::Mat<f32> {
    let source = [
        (-margin, -margin),
        (-margin, side_bits as f32 - 1.0 + margin),
        (
            side_bits as f32 - 1.0 + margin,
            side_bits as f32 - 1.0 + margin,
        ),
        (side_bits as f32 - 1.0 + margin, -margin),
    ];
    let mut mat_a = faer::Mat::<f32>::zeros(8, 9);
    for p in 0..4 {
        unsafe {
            mat_a.write_unchecked(p * 2, 0, source[p].0);
            mat_a.write_unchecked(p * 2, 1, source[p].1);
            mat_a.write_unchecked(p * 2, 2, 1.0);
            mat_a.write_unchecked(p * 2, 6, -1.0 * corners[p].0 * source[p].0);
            mat_a.write_unchecked(p * 2, 7, -1.0 * corners[p].0 * source[p].1);
            mat_a.write_unchecked(p * 2, 8, -1.0 * corners[p].0);
            mat_a.write_unchecked(p * 2 + 1, 3, source[p].0);
            mat_a.write_unchecked(p * 2 + 1, 4, source[p].1);
            mat_a.write_unchecked(p * 2 + 1, 5, 1.0);
            mat_a.write_unchecked(p * 2 + 1, 6, -1.0 * corners[p].1 * source[p].0);
            mat_a.write_unchecked(p * 2 + 1, 7, -1.0 * corners[p].1 * source[p].1);
            mat_a.write_unchecked(p * 2 + 1, 8, -1.0 * corners[p].1);
        }
    }
    // let svd = (mat_a.transpose()*mat_a.clone()).svd();
    let svd = mat_a.clone().svd();
    // println!("faer v {:?}", svd.v());
    let h = svd.v().col(8);
    faer::mat![[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]],]
}

pub fn hessian_response(img: &GrayImagef32) -> GrayImagef32 {
    let mut out = GrayImagef32::new(img.width(), img.height());
    for r in 1..(img.height() - 1) {
        for c in 1..(img.width() - 1) {
            let (v11, v12, v13, v21, v22, v23, v31, v32, v33) = unsafe {
                (
                    img.unsafe_get_pixel(c - 1, r - 1).0[0],
                    img.unsafe_get_pixel(c, r - 1).0[0],
                    img.unsafe_get_pixel(c + 1, r - 1).0[0],
                    img.unsafe_get_pixel(c - 1, r).0[0],
                    img.unsafe_get_pixel(c, r).0[0],
                    img.unsafe_get_pixel(c + 1, r).0[0],
                    img.unsafe_get_pixel(c - 1, r + 1).0[0],
                    img.unsafe_get_pixel(c, r + 1).0[0],
                    img.unsafe_get_pixel(c + 1, r + 1).0[0],
                )
            };
            let lxx = v21 - 2.0 * v22 + v23;
            let lyy = v12 - 2.0 * v22 + v32;
            let lxy = (v13 - v11 + v31 - v33) / 4.0;

            /* normalize and write out */
            unsafe {
                out.unsafe_put_pixel(c, r, [(lxx * lyy - lxy * lxy)].into());
            }
        }
    }
    out
}

pub fn pixel_bfs(
    mat: &mut GrayImagef32,
    cluster: &mut Vec<(u32, u32)>,
    x: u32,
    y: u32,
    threshold: f32,
) {
    if x < mat.width() && y < mat.height() {
        let v = unsafe { mat.unsafe_get_pixel(x, y).0[0] };
        if v < threshold {
            cluster.push((x, y));
            unsafe {
                mat.unsafe_put_pixel(x, y, [f32::MAX].into());
            }
            pixel_bfs(mat, cluster, x - 1, y, threshold);
            pixel_bfs(mat, cluster, x + 1, y, threshold);
            pixel_bfs(mat, cluster, x, y + 1, threshold);
            pixel_bfs(mat, cluster, x, y + 1, threshold);
        }
    }
}
