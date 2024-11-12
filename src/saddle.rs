use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct Saddle {
    pub p: (f32, f32),
    pub k: f32,
    pub theta: f32,
    pub phi: f32,
}

impl Saddle {
    pub fn arr(&self) -> [f32; 2] {
        [self.p.0, self.p.1]
    }
}

pub fn is_valid_quad(s0: &Saddle, d0: &Saddle, s1: &Saddle, d1: &Saddle) -> bool {
    if crate::math_util::theta_distance_degree(d0.theta, d1.theta) > 5.0 {
        // println!("f0");
        return false;
    }
    let v01 = (d0.p.0 - s0.p.0, d0.p.1 - s0.p.1);
    let v03 = (d1.p.0 - s0.p.0, d1.p.1 - s0.p.1);
    let v02 = (s1.p.0 - s0.p.0, s1.p.1 - s0.p.1);

    // filter white block
    let s0_theta = s0.theta / 180.0 * PI;
    let v_theta = (s0_theta.cos(), s0_theta.sin());
    let angle = crate::math_util::angle_degree(&v02, &v_theta).abs();

    if angle < 60.0 || angle > 120.0 {
        // println!("v02x {}", v02x);
        // println!("v02 {:?}", v02);
        // println!("x0 {:?}", x0);
        // println!("s0 {:?}", s0);
        // println!("f1");
        return false;
    }

    let c0 = crate::math_util::cross(&v01, &v02);
    let c1 = crate::math_util::cross(&v02, &v03);
    if c0 * c1 < 0.0 {
        // println!("f2");
        return false;
    }
    let v12 = (s1.p.0 - d0.p.0, s1.p.1 - d0.p.1);
    let v23 = (d1.p.0 - s1.p.0, d1.p.1 - s1.p.1);
    let c01 = crate::math_util::cross(&v01, &v12);
    let c12 = crate::math_util::cross(&v12, &v23);
    if c01 * c12 < 0.0 {
        // println!("f3");
        return false;
    }
    let v30 = (s0.p.0 - d1.p.0, s0.p.1 - d1.p.1);
    let a0 = crate::math_util::angle_degree(&v01, &v12);
    let a1 = crate::math_util::angle_degree(&v12, &v23);
    let a2 = crate::math_util::angle_degree(&v23, &v30);
    let a3 = crate::math_util::angle_degree(&v30, &v01);
    if (a0 - a2).abs() > 10.0 || (a1 - a3).abs() > 10.0 {
        // println!("f4");
        return false;
    }
    if crate::math_util::dot(&v01, &v02) < 0.0 || crate::math_util::dot(&v03, &v02) < 0.0 {
        return false;
    }
    true
}

pub fn saddle_distance2(s0: &Saddle, s1: &Saddle) -> f32 {
    let x = s0.p.0 - s1.p.0;
    let y = s0.p.1 - s1.p.1;
    x * x + y * y
}
