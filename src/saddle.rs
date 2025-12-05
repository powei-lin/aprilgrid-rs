use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct Saddle {
    pub p: (f32, f32),
    pub k: f32,
    pub theta: f32,
    pub phi: f32,
}

impl Saddle {
    pub const fn arr(&self) -> [f32; 2] {
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

    if !(60.0..=120.0).contains(&angle) {
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

pub const fn saddle_distance2(s0: &Saddle, s1: &Saddle) -> f32 {
    let x = s0.p.0 - s1.p.0;
    let y = s0.p.1 - s1.p.1;
    x * x + y * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saddle_distance2() {
        let s0 = Saddle {
            p: (0.0, 0.0),
            k: 0.0,
            theta: 0.0,
            phi: 0.0,
        };
        let s1 = Saddle {
            p: (3.0, 4.0),
            k: 0.0,
            theta: 0.0,
            phi: 0.0,
        };
        assert!((saddle_distance2(&s0, &s1) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_valid_quad() {
        // Construct a valid quad
        // s0 -- v01 --> d0 -- v12 --> s1 -- v23 --> d1 -- v30 --> s0
        // s0=(0,0), d0=(10,0), s1=(10,10), d1=(0,10)
        // theta needs to be set appropriately for angle checks.
        // s0 theta should align roughly with v01?
        // The code checks:
        // theta_distance_degree(d0.theta, d1.theta) > 5.0 => false
        // angle between v02 and s0.theta vector in [60, 120]

        let s0 = Saddle {
            p: (0.0, 0.0),
            k: 0.0,
            theta: 45.0,
            phi: 0.0,
        };
        let d0 = Saddle {
            p: (10.0, 0.0),
            k: 0.0,
            theta: 0.0,
            phi: 0.0,
        };
        let s1 = Saddle {
            p: (10.0, 10.0),
            k: 0.0,
            theta: 0.0,
            phi: 0.0,
        };
        let d1 = Saddle {
            p: (0.0, 10.0),
            k: 0.0,
            theta: 0.0,
            phi: 0.0,
        };

        // This simple setup might fail some checks, let's see.
        // d0.theta=0, d1.theta=0 -> diff=0 < 5. OK.
        // s0.theta=45. v=(0.707, 0.707). v02=(10, 10) -> (0.707, 0.707).
        // angle between v02 and v_theta is 0.
        // 0 is not in [60, 120]. So this will fail.

        // Let's try to make it fail first.
        assert!(!is_valid_quad(&s0, &d0, &s1, &d1));

        // Now try to make a valid one.
        // v02 is diagonal (1,1). We want s0.theta to be perpendicular to it?
        // If angle is 90, then s0.theta should be -45 or 135.
        let s0_valid = Saddle {
            p: (0.0, 0.0),
            k: 0.0,
            theta: 135.0,
            phi: 0.0,
        };
        // v_theta = (-0.7, 0.7). v02=(10,10). dot product = 0. angle 90. OK.

        // Cross products:
        // v01=(10,0), v02=(10,10). cross = 100 > 0.
        // v02=(10,10), v03=(0,10). cross = 100 > 0.
        // c0*c1 > 0. OK.

        // v12 = s1-d0 = (0,10). v01=(10,0). cross = 100.
        // v23 = d1-s1 = (-10,0). v12=(0,10). cross = 100.
        // c01*c12 > 0. OK.

        // Angles:
        // a0 (v01, v12) = 90
        // a1 (v12, v23) = 90
        // a2 (v23, v30) = 90. v30=s0-d1=(0,-10).
        // a3 (v30, v01) = 90.
        // |a0-a2| = 0. |a1-a3| = 0. OK.

        // Dots:
        // v01.v02 = 100 > 0.
        // v03.v02 = 100 > 0.

        assert!(is_valid_quad(&s0_valid, &d0, &s1, &d1));
    }
}
