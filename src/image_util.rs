use faer::linalg::solvers::SolveLstsqCore;
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
            *mat_a.get_mut_unchecked(p * 2, 0) = source[p].0;
            *mat_a.get_mut_unchecked(p * 2, 1) = source[p].1;
            *mat_a.get_mut_unchecked(p * 2, 2) = 1.0;
            *mat_a.get_mut_unchecked(p * 2, 6) = -corners[p].0 * source[p].0;
            *mat_a.get_mut_unchecked(p * 2, 7) = -corners[p].0 * source[p].1;
            *mat_a.get_mut_unchecked(p * 2, 8) = -corners[p].0;
            *mat_a.get_mut_unchecked(p * 2 + 1, 3) = source[p].0;
            *mat_a.get_mut_unchecked(p * 2 + 1, 4) = source[p].1;
            *mat_a.get_mut_unchecked(p * 2 + 1, 5) = 1.0;
            *mat_a.get_mut_unchecked(p * 2 + 1, 6) = -corners[p].1 * source[p].0;
            *mat_a.get_mut_unchecked(p * 2 + 1, 7) = -corners[p].1 * source[p].1;
            *mat_a.get_mut_unchecked(p * 2 + 1, 8) = -corners[p].1;
        }
    }
    // let svd = (mat_a.transpose()*mat_a.clone()).svd();
    let svd = mat_a.clone().svd().expect("tag svd failed");
    // println!("faer v {:?}", svd.v());
    let h = svd.V().col(8);
    faer::mat![[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]],]
}

pub fn tag_affine(corners: &[(f32, f32)], side_bits: u8, margin: f32) -> faer::Mat<f32> {
    let source = [
        (-margin, -margin),
        (-margin, side_bits as f32 - 1.0 + margin),
        (
            side_bits as f32 - 1.0 + margin,
            side_bits as f32 - 1.0 + margin,
        ),
        (side_bits as f32 - 1.0 + margin, -margin),
    ];

    let mut mat_a: faer::Mat<f32> = faer::Mat::zeros(8, 6);
    let mut mat_b: faer::Mat<f32> = faer::Mat::zeros(8, 1);

    for p in 0..4 {
        unsafe {
            *mat_a.get_mut_unchecked(p * 2, 0) = source[p].0;
            *mat_a.get_mut_unchecked(p * 2, 1) = source[p].1;
            *mat_a.get_mut_unchecked(p * 2, 2) = 1.0;
            *mat_a.get_mut_unchecked(p * 2 + 1, 3) = source[p].0;
            *mat_a.get_mut_unchecked(p * 2 + 1, 4) = source[p].1;
            *mat_a.get_mut_unchecked(p * 2 + 1, 5) = 1.0;
            *mat_b.get_mut_unchecked(p * 2, 0) = corners[p].0;
            *mat_b.get_mut_unchecked(p * 2 + 1, 0) = corners[p].1;
        }
    }
    mat_a
        .qr()
        .solve_lstsq_in_place_with_conj(faer::Conj::No, mat_b.as_mut());
    let h = mat_b.col(0);
    faer::mat![[h[0], h[1], h[2]], [h[3], h[4], h[5]], [0.0, 0.0, 1.0],]
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
    let mut stack = vec![(x, y)];
    while let Some((cx, cy)) = stack.pop() {
        if cx >= mat.width() || cy >= mat.height() {
            continue;
        }
        let v = unsafe { mat.unsafe_get_pixel(cx, cy).0[0] };
        if v < threshold {
            cluster.push((cx, cy));
            unsafe {
                mat.unsafe_put_pixel(cx, cy, [f32::MAX].into());
            }
            if cx > 0 {
                stack.push((cx - 1, cy));
            }
            stack.push((cx + 1, cy));
            if cy > 0 {
                stack.push((cx, cy - 1));
            }
            stack.push((cx, cy + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    #[test]
    fn test_tag_homography() {
        // Simple case: identity mapping (with some scaling/translation)
        // corners: (0,0), (0,10), (10,10), (10,0)
        // side_bits: 10
        // margin: 0
        // source: (0,0), (0,9), (9,9), (9,0)
        // This should result in a homography that maps source to corners.
        // corners are roughly same as source but scaled by 10/9.
        let corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let h = tag_homography(&corners, 10, 0.0);
        assert!(h.nrows() == 3);
        assert!(h.ncols() == 3);
        // We don't easily check values without applying it, but it shouldn't panic.
    }

    #[test]
    fn test_tag_affine() {
        let corners = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let h = tag_affine(&corners, 10, 0.0);
        assert!(h.nrows() == 3);
        assert!(h.ncols() == 3);
        assert!((h[(2, 0)] - 0.0).abs() < 1e-6);
        assert!((h[(2, 1)] - 0.0).abs() < 1e-6);
        assert!((h[(2, 2)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hessian_response() {
        let mut img = GrayImagef32::new(5, 5);
        // Set center pixel high, neighbors low -> high curvature
        for y in 0..5 {
            for x in 0..5 {
                img.put_pixel(x, y, Luma([0.0]));
            }
        }
        img.put_pixel(2, 2, Luma([10.0]));

        // Hessian response should be non-zero around center
        let resp = hessian_response(&img);
        // Center (2,2) response depends on neighbors.
        // At (2,2): v22=10, others=0.
        // lxx = 0 - 20 + 0 = -20
        // lyy = 0 - 20 + 0 = -20
        // lxy = 0
        // det = 400
        // But the function iterates 1..h-1, 1..w-1.
        // So (2,2) is computed.
        let val = resp.get_pixel(2, 2)[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_pixel_bfs() {
        let mut img = GrayImagef32::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                img.put_pixel(x, y, Luma([100.0]));
            }
        }
        // Create a dark region
        img.put_pixel(2, 2, Luma([10.0]));
        img.put_pixel(2, 3, Luma([10.0]));

        let mut cluster = Vec::new();
        pixel_bfs(&mut img, &mut cluster, 2, 2, 50.0);

        assert_eq!(cluster.len(), 2);
        assert!(cluster.contains(&(2, 2)));
        assert!(cluster.contains(&(2, 3)));

        // Visited pixels should be set to MAX
        assert_eq!(img.get_pixel(2, 2)[0], f32::MAX);
        assert_eq!(img.get_pixel(2, 3)[0], f32::MAX);
    }
}
