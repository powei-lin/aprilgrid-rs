use faer;
use nalgebra as na;
pub fn tag_homography(corners: &[(f32, f32)], side_bits: u8) -> faer::Mat<f32> {
    let margin = 0.3;
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

pub fn tag_homography_na(corners: &[(f32, f32)], side_bits: u8) -> na::SMatrix<f32, 3, 3> {
    let margin = 0.3;
    let source = [
        (-margin, -margin),
        (-margin, side_bits as f32 - 1.0 + margin),
        (
            side_bits as f32 - 1.0 + margin,
            side_bits as f32 - 1.0 + margin,
        ),
        (side_bits as f32 - 1.0 + margin, -margin),
    ];
    // let mut mat_a = faer::Mat::<f32>::zeros(8, 9);
    let mut mat_a = na::SMatrix::<f32, 8, 9>::zeros();

    for p in 0..4 {
        mat_a[(p * 2, 0)] = source[p].0;
        mat_a[(p * 2, 1)] = source[p].1;
        mat_a[(p * 2, 2)] = 1.0;
        mat_a[(p * 2, 6)] = -1.0 * corners[p].0 * source[p].0;
        mat_a[(p * 2, 7)] = -1.0 * corners[p].0 * source[p].1;
        mat_a[(p * 2, 8)] = -1.0 * corners[p].0;
        mat_a[(p * 2 + 1, 3)] = source[p].0;
        mat_a[(p * 2 + 1, 4)] = source[p].1;
        mat_a[(p * 2 + 1, 5)] = 1.0;
        mat_a[(p * 2 + 1, 6)] = -1.0 * corners[p].1 * source[p].0;
        mat_a[(p * 2 + 1, 7)] = -1.0 * corners[p].1 * source[p].1;
        mat_a[(p * 2 + 1, 8)] = -1.0 * corners[p].1;
    }
    // let svd = (mat_a.transpose()*mat_a.clone()).svd();
    // let v: na::SMatrix<f32, 8, 9> = mat_a.svd(false, true).v_t.unwrap();
    let v: na::SMatrix<f32, 8, 9> = mat_a.svd(true, true).v_t.unwrap();
    println!("v {}", v);
    // let svd = mat_a.clone().svd();
    let h: na::SMatrix<f32, 3, 3> = v
        .row(7)
        .clone_owned()
        .reshape_generic(na::Const::<3>, na::Const::<3>);
    // faer::mat![[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]],]
    h
}
