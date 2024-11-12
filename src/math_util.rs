use std::f32::consts::PI;

use faer::solvers::SpSolver;

pub fn find_xy(a0: f32, b0: f32, c0: f32, a1: f32, b1: f32, c1: f32) -> (f32, f32) {
    let a = faer::mat![[a0, b0], [a1, b1]];
    let b = faer::mat![[-c0], [-c1]];
    let plu = a.partial_piv_lu();
    let x1 = plu.solve(&b);

    unsafe { (*x1.get_unchecked(0, 0), *x1.get_unchecked(1, 0)) }
}

/// Abs theta diff [0, pi/2]
pub fn theta_distance(t0: f32, t1: f32) -> f32 {
    let mut d = t0 - t1 + PI / 2.0;
    if d < 0.0 {
        d += PI;
    } else if d > PI {
        d -= PI;
    }
    (d - PI / 2.0).abs()
}
pub fn cross(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    v0.0 * v1.1 - v0.1 * v1.0
}
pub fn dot(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    v0.0 * v1.0 + v0.1 * v1.1
}

pub fn angle_degree(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    (v1.1 * v0.0 - v1.0 * v0.1).atan2(v0.0 * v1.0 + v0.1 * v1.1) * 180.0 / PI
}
