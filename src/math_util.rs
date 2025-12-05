use std::f32::consts::PI;

use faer::linalg::solvers::Solve;

pub fn find_xy(a0: f32, b0: f32, c0: f32, a1: f32, b1: f32, c1: f32) -> (f32, f32) {
    let a = faer::mat![[a0, b0], [a1, b1]];
    let b = faer::mat![[-c0], [-c1]];
    let plu = a.partial_piv_lu();
    let x1 = plu.solve(&b);

    unsafe { (*x1.get_unchecked(0, 0), *x1.get_unchecked(1, 0)) }
}

/// Abs theta diff [0, 90.0]
pub const fn theta_distance_degree(t0: f32, t1: f32) -> f32 {
    let mut d = t0 - t1 + 90.0;
    if d < 0.0 {
        d += 180.0;
    } else if d > 180.0 {
        d -= 180.0;
    }
    if d > 90.0 { d - 90.0 } else { 90.0 - d }
}
pub const fn cross(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    v0.0 * v1.1 - v0.1 * v1.0
}
pub const fn dot(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    v0.0 * v1.0 + v0.1 * v1.1
}

pub fn angle_degree(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    (v1.1 * v0.0 - v1.0 * v0.1).atan2(v0.0 * v1.0 + v0.1 * v1.1) * 180.0 / PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_xy() {
        // x + y = 2
        // x - y = 0
        // => x = 1, y = 1
        let (x, y) = find_xy(1.0, 1.0, -2.0, 1.0, -1.0, 0.0);
        assert!((x - 1.0).abs() < 1e-6);
        assert!((y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_theta_distance_degree() {
        assert!((theta_distance_degree(0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((theta_distance_degree(0.0, 90.0) - 90.0).abs() < 1e-6);
        assert!((theta_distance_degree(0.0, 45.0) - 45.0).abs() < 1e-6);
        assert!((theta_distance_degree(0.0, 180.0) - 0.0).abs() < 1e-6); // 180 is same as 0 mod 180 for lines? No, it's [0, 90] diff
        // Let's check implementation:
        // d = t0 - t1 + 90.0
        // if d < 0 => d += 180
        // if d > 180 => d -= 180
        // if d > 90 => d - 90 else 90 - d
        // t0=0, t1=180 => d = -90 + 90 = -90 => d+=180 => 90. d>90? No. 90-90=0. Correct.

        assert!((theta_distance_degree(10.0, 20.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross() {
        let v0 = (1.0, 0.0);
        let v1 = (0.0, 1.0);
        assert!((cross(&v0, &v1) - 1.0).abs() < 1e-6);
        assert!((cross(&v1, &v0) - -1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot() {
        let v0 = (1.0, 0.0);
        let v1 = (0.0, 1.0);
        assert!((dot(&v0, &v1) - 0.0).abs() < 1e-6);
        let v2 = (1.0, 1.0);
        assert!((dot(&v0, &v2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_angle_degree() {
        let v0 = (1.0, 0.0);
        let v1 = (0.0, 1.0);
        assert!((angle_degree(&v0, &v1) - 90.0).abs() < 1e-6);
        let v2 = (1.0, 1.0);
        assert!((angle_degree(&v0, &v2) - 45.0).abs() < 1e-6);
    }
}
