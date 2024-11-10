use core::f32;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    ops::BitXor,
};

use crate::{homography, quad, tag_families};
use faer::solvers::SpSolverLstsq;
use image::{DynamicImage, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use itertools::Itertools;

type GrayImagef32 = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;

pub struct TagDetector {
    edge: u8,
    border: u8,
    hamming_distance: u8,
    code_list: Vec<u64>,
    detector_params: DetectorParams,
}

pub struct DetectorParams {
    pub brightness_mean_value: u8,
    pub quad_corner_compensate_pixel: f32,
    pub margin: f32,
}

impl DetectorParams {
    pub fn default_params() -> DetectorParams {
        DetectorParams {
            brightness_mean_value: 100,
            quad_corner_compensate_pixel: 2.5,
            margin: 0.3,
        }
    }
}
pub fn decode_positions(
    img_w: u32,
    img_h: u32,
    quad_pts: &[(f32, f32)],
    border_bits: u8,
    edge_bits: u8,
    margin: f32,
) -> Option<Vec<(f32, f32)>> {
    if quad_pts.iter().any(|(x, y)| {
        let x = x.round() as u32;
        let y = y.round() as u32;
        x >= img_w || y >= img_h
    }) {
        return None;
    }
    let side_bits = border_bits * 2 + edge_bits;
    let h = homography::tag_homography(quad_pts, side_bits, margin);
    Some(
        (border_bits..border_bits + edge_bits)
            .flat_map(|x| {
                (border_bits..border_bits + edge_bits)
                    .map(|y| {
                        let tp = faer::mat![[x as f32], [y as f32], [1.0]];
                        let tt = h.clone() * tp;
                        (tt[(0, 0)] / tt[(2, 0)], tt[(1, 0)] / tt[(2, 0)])
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

pub fn bit_code(
    img: &GrayImage,
    decode_pts: &[(f32, f32)],
    valid_brightness_threshold: u8,
    max_invalid_bit: u32,
) -> Option<u64> {
    let brightness_vec: Vec<u8> = decode_pts
        .iter()
        .filter_map(|(x, y)| {
            let (x, y) = (x.round() as u32, y.round() as u32);
            if x >= img.width() || y >= img.height() {
                None
            } else {
                Some(img.get_pixel(x, y).0[0])
            }
        })
        .collect();
    if brightness_vec.len() != decode_pts.len() {
        return None;
    }
    let (min_b, max_b) = brightness_vec
        .iter()
        .fold((255, 0), |(min_b, max_b), e| (min_b.min(*e), max_b.max(*e)));
    if max_b - min_b < 50 {
        return None;
    }
    let mid_b = ((min_b as f32 + max_b as f32) / 2.0).round() as u8;
    let (bits, invalid_count): (u64, u32) = brightness_vec.iter().rev().enumerate().fold(
        (0u64, 0u32),
        |(acc, invalid_count), (i, b)| {
            if (mid_b as i32 - *b as i32).abs() < valid_brightness_threshold as i32 {
                (acc, invalid_count + 1)
            } else if *b > mid_b {
                (acc | (1 << i), invalid_count)
            } else {
                (acc, invalid_count)
            }
        },
    );
    if invalid_count > max_invalid_bit {
        None
    } else {
        Some(bits)
    }
}

fn rotate_bits(bits: u64, edge_bits: u8) -> u64 {
    let edge_bits = edge_bits as usize;
    let mut b = 0u64;
    let mut count = 0;
    for r in (0..edge_bits).rev() {
        for c in 0..edge_bits {
            let idx = r + c * edge_bits;
            b |= ((bits >> idx) & 1) << count;
            count += 1
        }
    }
    b
}

pub fn best_tag(bits: u64, thres: u8, tag_family: &[u64], edge_bits: u8) -> Option<(usize, usize)> {
    let mut bits = bits;
    for rotated in 0..4 {
        let scores: Vec<u32> = tag_family
            .iter()
            .map(|t| t.bitxor(bits).count_ones())
            .collect();
        let (best_idx, best_score) = scores
            .iter()
            .enumerate()
            .reduce(|(best_idx, best_score), (cur_idx, cur_socre)| {
                if cur_socre < best_score {
                    (cur_idx, cur_socre)
                } else {
                    (best_idx, best_score)
                }
            })
            .unwrap();
        if *best_score < thres as u32 {
            // println!("best {} {} rotate {}", best_idx, best_score, rotated);
            return Some((best_idx, rotated));
        } else if rotated == 3 {
            break;
        }
        bits = rotate_bits(bits, edge_bits);
    }
    None
}

fn hessian_response(img: &GrayImagef32) -> GrayImagef32 {
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

fn pixel_bfs(
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

fn init_saddle_clusters(h_mat: &GrayImagef32, threshold: f32) -> Vec<Vec<(u32, u32)>> {
    let mut tmp_h_mat = h_mat.clone();
    let mut clusters = Vec::new();
    for r in 1..h_mat.height() - 1 {
        for c in 1..h_mat.width() - 1 {
            let mut cluster = Vec::new();
            pixel_bfs(&mut tmp_h_mat, &mut cluster, c, r, threshold);
            if cluster.len() > 0 {
                clusters.push(cluster);
            }
        }
    }
    clusters
}

pub fn rochade_refine<T>(
    image_input: &ImageBuffer<Luma<T>, Vec<T>>,
    initial_corners: &Vec<(f32, f32)>,
    half_size_patch: i32,
) -> Option<Vec<(f32, f32)>>
where
    T: image::Primitive + Into<f32>,
{
    let mut refined_corners = Vec::<(f32, f32)>::new();
    // kernel
    let kernel_size = (half_size_patch * 2 + 1) as usize;
    let gamma = half_size_patch as f32;
    let flat_k_slice: Vec<f32> = (0..kernel_size)
        .flat_map(|i| {
            (0..kernel_size)
                .map(move |j| {
                    0.0_f32.max(
                        gamma + 1.0
                            - ((gamma - i as f32) * (gamma - i as f32)
                                + (gamma - j as f32) * (gamma - j as f32))
                                .sqrt(),
                    )
                })
                .collect::<Vec<f32>>()
        })
        .collect();
    let s = flat_k_slice.iter().sum::<f32>();
    let flat_k: Vec<f32> = flat_k_slice.iter().map(|v| v / s).collect();

    let (width, height) = (image_input.width() as i32, image_input.height() as i32);
    let half_size_patch2 = half_size_patch * 2;

    // iter all corner
    for &(initial_x, initial_y) in initial_corners {
        let round_x = initial_x.round() as i32;
        let round_y = initial_y.round() as i32;
        if (round_y - half_size_patch2) < 0
            || (round_y + half_size_patch2 >= height)
            || (round_x - half_size_patch2 < 0)
            || (round_x + half_size_patch2 >= width)
        {
            return None;
        }

        // patch
        let patch_size: usize = 4 * half_size_patch as usize + 1;
        let patch = image_input.view(
            (round_x - half_size_patch2) as u32,
            (round_y - half_size_patch2) as u32,
            (patch_size) as u32,
            (patch_size) as u32,
        );

        let mut smooth_sub_image: faer::Mat<f32> = faer::Mat::zeros(kernel_size, kernel_size);
        for r in 0..kernel_size {
            for c in 0..kernel_size {
                let sub_patch_vec: Vec<f32> = patch
                    .view(c as u32, r as u32, kernel_size as u32, kernel_size as u32)
                    .pixels()
                    .map(|(_, _, v)| v.0[0].into())
                    .collect();
                let conv_p = sub_patch_vec
                    .iter()
                    .zip(&flat_k)
                    .fold(0.0_f32, |acc, (k, p)| acc + k * p);
                smooth_sub_image[(r, c)] = conv_p;
            }
        }

        // a_1*x^2 + a_2*x*y + a_3*y^2 + a_4*x + a_5*y + a_6 = f
        let mut mat_a: faer::Mat<f32> = faer::Mat::ones(kernel_size * kernel_size, 6);
        let mut mat_b: faer::Mat<f32> = faer::Mat::zeros(kernel_size * kernel_size, 1);
        let mut count = 0;
        for r in 0..kernel_size {
            for c in 0..kernel_size {
                let x = c as f32 - half_size_patch as f32;
                let y = r as f32 - half_size_patch as f32;
                let f = smooth_sub_image[(r, c)];
                mat_a[(count, 0)] = x * x;
                mat_a[(count, 1)] = x * y;
                mat_a[(count, 2)] = y * y;
                mat_a[(count, 3)] = x;
                mat_a[(count, 4)] = y;
                mat_b[(count, 0)] = f;
                count += 1;
            }
        }
        let params = mat_a.qr().solve_lstsq(mat_b);
        // println!("conv {:?}", smooth_sub_image);
        // println!("params {:?}", params);
        let a1 = params[(0, 0)];
        let a2 = params[(1, 0)];
        let a3 = params[(2, 0)];
        let a4 = params[(3, 0)];
        let a5 = params[(4, 0)];
        // let a6 = params[(5, 0)];
        let fxx = 2.0 * a1;
        let fyy = 2.0 * a3;
        let fxy = a2;
        let d = fxx * fyy - fxy * fxy;
        // is saddle point
        if d < 0.0 {
            let (x0, y0) = quad::find_xy(2.0 * a1, a2, a4, a2, 2.0 * a3, a5);
            if x0.abs() > half_size_patch as f32 || y0.abs() > half_size_patch as f32 {
                return None;
            } else {
                refined_corners.push((initial_x.round() + x0, initial_y.round() + y0));
            }
        } else {
            return None;
        }
    }

    Some(refined_corners)
}

fn saddle_distance2(s0: &Saddle, s1: &Saddle) -> f32 {
    let x = s0.p.0 - s1.p.0;
    let y = s0.p.1 - s1.p.1;
    x * x + y * y
}

fn closest_n_idx(
    saddles: &[Saddle],
    self_idx: usize,
    active_idxs: &HashSet<usize>,
    num: usize,
) -> Vec<usize> {
    let mut sorted: Vec<_> = saddles
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if active_idxs.contains(&i) {
                Some((i, s.clone()))
            } else {
                None
            }
        })
        .collect();
    let target = saddles[self_idx];
    sorted.sort_by(|(_, a), (_, b)| {
        saddle_distance2(&target, a)
            .partial_cmp(&saddle_distance2(&target, b))
            .unwrap()
    });
    let out_len = sorted.len().min(num);
    sorted[0..out_len].iter().map(|(i, _)| *i).collect()
}

fn cross(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    v0.0 * v1.1 - v0.1 * v1.0
}
fn dot(v0: &(f32, f32), v1: &(f32, f32)) -> f32 {
    v0.0 * v1.0 + v0.1 * v1.1
}

#[derive(Debug, Clone, Copy)]
pub struct Saddle {
    pub p: (f32, f32),
    pub k: f32,
    pub theta: f32,
    pub phi: f32,
}

pub fn rochade_refine2<T>(
    image_input: &ImageBuffer<Luma<T>, Vec<T>>,
    initial_corners: &Vec<(f32, f32)>,
    half_size_patch: i32,
) -> Vec<Saddle>
where
    T: image::Primitive + Into<f32>,
{
    // let mut refined_corners = Vec::<(f32, f32)>::new();
    let mut refined_corners = Vec::<Saddle>::new();
    // kernel
    let kernel_size = (half_size_patch * 2 + 1) as usize;
    let gamma = half_size_patch as f32;
    let flat_k_slice: Vec<f32> = (0..kernel_size)
        .flat_map(|i| {
            (0..kernel_size)
                .map(move |j| {
                    0.0_f32.max(
                        gamma + 1.0
                            - ((gamma - i as f32) * (gamma - i as f32)
                                + (gamma - j as f32) * (gamma - j as f32))
                                .sqrt(),
                    )
                })
                .collect::<Vec<f32>>()
        })
        .collect();
    let s = flat_k_slice.iter().sum::<f32>();
    let flat_k: Vec<f32> = flat_k_slice.iter().map(|v| v / s).collect();

    let (width, height) = (image_input.width() as i32, image_input.height() as i32);
    let half_size_patch2 = half_size_patch * 2;

    // iter all corner
    for &(initial_x, initial_y) in initial_corners {
        let round_x = initial_x.round() as i32;
        let round_y = initial_y.round() as i32;
        if (round_y - half_size_patch2) < 0
            || (round_y + half_size_patch2 >= height)
            || (round_x - half_size_patch2 < 0)
            || (round_x + half_size_patch2 >= width)
        {
            continue;
        }

        // patch
        let patch_size: usize = 4 * half_size_patch as usize + 1;
        let patch = image_input.view(
            (round_x - half_size_patch2) as u32,
            (round_y - half_size_patch2) as u32,
            (patch_size) as u32,
            (patch_size) as u32,
        );

        let mut smooth_sub_image: faer::Mat<f32> = faer::Mat::zeros(kernel_size, kernel_size);
        for r in 0..kernel_size {
            for c in 0..kernel_size {
                let sub_patch_vec: Vec<f32> = patch
                    .view(c as u32, r as u32, kernel_size as u32, kernel_size as u32)
                    .pixels()
                    .map(|(_, _, v)| v.0[0].into())
                    .collect();
                let conv_p = sub_patch_vec
                    .iter()
                    .zip(&flat_k)
                    .fold(0.0_f32, |acc, (k, p)| acc + k * p);
                smooth_sub_image[(r, c)] = conv_p;
            }
        }

        // a_1*x^2 + a_2*x*y + a_3*y^2 + a_4*x + a_5*y + a_6 = f
        let mut mat_a: faer::Mat<f32> = faer::Mat::ones(kernel_size * kernel_size, 6);
        let mut mat_b: faer::Mat<f32> = faer::Mat::zeros(kernel_size * kernel_size, 1);
        let mut count = 0;
        for r in 0..kernel_size {
            for c in 0..kernel_size {
                let x = c as f32 - half_size_patch as f32;
                let y = r as f32 - half_size_patch as f32;
                let f = smooth_sub_image[(r, c)];
                mat_a[(count, 0)] = x * x;
                mat_a[(count, 1)] = x * y;
                mat_a[(count, 2)] = y * y;
                mat_a[(count, 3)] = x;
                mat_a[(count, 4)] = y;
                mat_b[(count, 0)] = f;
                count += 1;
            }
        }
        let params = mat_a.qr().solve_lstsq(mat_b);
        // println!("conv {:?}", smooth_sub_image);
        // println!("params {:?}", params);
        let a1 = params[(0, 0)];
        let a2 = params[(1, 0)];
        let a3 = params[(2, 0)];
        let a4 = params[(3, 0)];
        let a5 = params[(4, 0)];
        // let a6 = params[(5, 0)];
        let fxx = 2.0 * a1;
        let fyy = 2.0 * a3;
        let fxy = a2;
        let d = fxx * fyy - fxy * fxy;

        // is saddle point
        if d < 0.0 {
            let (x0, y0) = quad::find_xy(2.0 * a1, a2, a4, a2, 2.0 * a3, a5);
            // move too much
            if x0.abs() > 1.0 || y0.abs() > 1.0 {
                // println!("moving {:0.4} {:0.4} {} {}", initial_x, initial_y, x0, y0);
                continue;
            } else {
                // Alturki, Abdulrahman S., and John S. Loomis.
                // "A new X-Corner Detection for Camera Calibration Using Saddle Points."
                let c5 = (a1 + a3) / 2.0;
                let c4 = (a1 - a3) / 2.0;
                let c3 = a2 / 2.0;
                let k = (c4 * c4 + c3 * c3).sqrt();
                let phi = (-1.0 * c5 / k).acos() / 2.0 / PI * 180.0;
                let theta = (c3 / k).asin() / 2.0 / PI * 180.0;
                refined_corners.push(Saddle {
                    p: (initial_x.round() + x0, initial_y.round() + y0),
                    k,
                    theta,
                    phi,
                });
            }
        } else {
            continue;
        }
    }
    refined_corners
}
impl TagDetector {
    pub fn new(
        tag_family: &tag_families::TagFamily,
        optional_detector_params: Option<DetectorParams>,
    ) -> TagDetector {
        let detector_params = optional_detector_params.unwrap_or(DetectorParams::default_params());
        match tag_family {
            tag_families::TagFamily::T16H5 => TagDetector {
                edge: 4,
                border: 2,
                hamming_distance: 1,
                code_list: tag_families::T16H5.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T25H7 => TagDetector {
                edge: 5,
                border: 2,
                hamming_distance: 2,
                code_list: tag_families::T25H7.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T25H9 => TagDetector {
                edge: 5,
                border: 2,
                hamming_distance: 3,
                code_list: tag_families::T25H9.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T36H11 => TagDetector {
                edge: 6,
                border: 2,
                hamming_distance: 3,
                code_list: tag_families::T36H11.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T36H11B1 => TagDetector {
                edge: 6,
                border: 1,
                hamming_distance: 3,
                code_list: tag_families::T36H11.to_vec(),
                detector_params,
            },
        }
    }
    pub fn detect(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let img_grey = img.to_luma8();
        let img_adjust_bri =
            quad::adjust_brightness(img, self.detector_params.brightness_mean_value);
        let quads = quad::find_quad(&img_adjust_bri, 400.0);
        let mut detected_tags = HashMap::<u32, [(f32, f32); 4]>::new();
        for c in quads.iter() {
            let homo_points_option = decode_positions(
                img.width(),
                img.height(),
                c,
                self.border,
                self.edge,
                self.detector_params.margin,
            );
            if let Some(homo_points) = homo_points_option {
                let bits = bit_code(&img_grey, &homo_points, 30, 5);
                if bits.is_none() {
                    continue;
                }
                let tag_id_option = best_tag(
                    bits.unwrap(),
                    self.hamming_distance,
                    &self.code_list,
                    self.edge,
                );
                if tag_id_option.is_none() {
                    continue;
                }
                let tag_id = tag_id_option.unwrap();

                // compensate dilation causing quad corner shift
                let quad_center = c
                    .iter()
                    .fold((0.0, 0.0), |acc, e| (acc.0 + e.0, acc.1 + e.1));
                let (qcx, qcy) = (quad_center.0 / 4.0, quad_center.1 / 4.0);

                let mut c: Vec<(f32, f32)> = c
                    .iter()
                    .map(|(qx, qy)| {
                        let (vx, vy) = (qx - qcx, qy - qcy);
                        let n = (vx * vx + vy * vy).sqrt();
                        let scale = self.detector_params.quad_corner_compensate_pixel;
                        (qx + vx / n * scale, qy + vy / n * scale)
                    })
                    .collect();
                c.rotate_left(tag_id.1);
                if let Some(refined) = rochade_refine(&img_adjust_bri.to_luma8(), &c, 4) {
                    let refined_arr: [(f32, f32); 4] = refined.try_into().unwrap();
                    detected_tags.insert(tag_id.0 as u32, refined_arr);
                }
            }
        }
        detected_tags
    }
    pub fn refined_saddle_points(&self, img: &DynamicImage) -> Vec<Saddle> {
        let blur: GrayImagef32 = imageproc::filter::gaussian_blur_f32(&img.to_luma32f(), 1.5);
        let hessian_response_mat = hessian_response(&blur);
        let min_response = hessian_response_mat
            .to_vec()
            .iter()
            .fold(f32::MAX, |acc, e| acc.min(*e));
        let min_response_threshold = min_response * 0.05;
        let saddle_clusters = init_saddle_clusters(&hessian_response_mat, min_response_threshold);
        let saddle_cluster_centers: Vec<(f32, f32)> = saddle_clusters
            .iter()
            .map(|c| {
                let (sx, sy) = c.iter().fold((0.0, 0.0), |(ax, ay), (ex, ey)| {
                    (ax + *ex as f32, ay + *ey as f32)
                });
                (sx / c.len() as f32, sy / c.len() as f32)
            })
            .collect();
        let saddle_points = rochade_refine2(&blur, &saddle_cluster_centers, 2);
        let smax = saddle_points.iter().fold(f32::MIN, |acc, s| acc.max(s.k)) / 10.0;
        let min_angle = 30.0;
        let max_angle = 60.0;
        let refined: Vec<Saddle> = saddle_points
            .iter()
            .filter_map(|s| {
                if s.k < smax || s.phi < min_angle || s.phi > max_angle {
                    None
                } else {
                    Some(s.to_owned())
                }
            })
            .collect();
        refined
    }
    pub fn detect2(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let mut detected_tags = HashMap::new();
        let mut avg_tag_l = Vec::new();
        let img_grey = img.to_luma8();
        let refined = self.refined_saddle_points(&img);
        if refined.len() < 4 {
            return detected_tags;
        }
        let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();

        let mut start_idx = active_idxs.iter().next().unwrap().clone();
        while active_idxs.len() >= 4 {
            if !active_idxs.remove(&start_idx) {
                start_idx = active_idxs.iter().next().unwrap().clone();
                continue;
            }
            let current_saddle = refined[start_idx];
            // println!("start idx: {} {} {}", start_idx, refined[start_idx].p.0, refined[start_idx].p.1);
            let closest_idxs = closest_n_idx(&refined, start_idx, &active_idxs, 50);

            let it = (0..closest_idxs.len()).combinations(3);
            for iv in it {
                let mut id_saddle: Vec<(usize, Saddle)> = iv
                    .iter()
                    .map(|i| (closest_idxs[*i], refined[closest_idxs[*i]]))
                    .collect();
                id_saddle.sort_by(|a, b| {
                    (current_saddle.theta - a.1.theta)
                        .abs()
                        .partial_cmp(&(current_saddle.theta - b.1.theta).abs())
                        .unwrap()
                });
                let (idx_i, cross_saddle) = id_saddle[0];
                let (idx_j, side_saddle0) = id_saddle[1];
                let (idx_k, side_saddle1) = id_saddle[2];
                // let relative_theta_sort = [i, j, k]
                if (current_saddle.theta - cross_saddle.theta).abs() > 10.0 {
                    continue;
                }
                if (side_saddle0.theta - side_saddle1.theta).abs() > 10.0 {
                    continue;
                }

                let l0 = saddle_distance2(&current_saddle, &side_saddle0).sqrt();
                let l1 = saddle_distance2(&current_saddle, &side_saddle1).sqrt();
                let l2 = saddle_distance2(&cross_saddle, &side_saddle0).sqrt();
                let l3 = saddle_distance2(&cross_saddle, &side_saddle1).sqrt();
                let avg_l = (l0 + l1 + l2 + l3) / 4.0;
                let l_ratio = 0.3;
                let min_l = avg_l * (1.0 - l_ratio);
                let max_l = avg_l * (1.0 + l_ratio);
                if avg_tag_l.len() > 4 {
                    let global_avg_l = avg_tag_l.iter().sum::<f32>() / avg_tag_l.len() as f32;
                    if avg_l < global_avg_l * 0.7 || avg_l > global_avg_l * 1.3 {
                        continue;
                    }
                }
                if l0 < min_l
                    || l0 > max_l
                    || l1 < min_l
                    || l1 > max_l
                    || l2 < min_l
                    || l2 > max_l
                    || l3 < min_l
                    || l3 > max_l
                {
                    continue;
                }
                let v0 = (
                    side_saddle0.p.0 - current_saddle.p.0,
                    side_saddle0.p.1 - current_saddle.p.1,
                );
                let v1 = (
                    side_saddle1.p.0 - current_saddle.p.0,
                    side_saddle1.p.1 - current_saddle.p.1,
                );
                let v2 = (
                    cross_saddle.p.0 - current_saddle.p.0,
                    cross_saddle.p.1 - current_saddle.p.1,
                );
                let c0 = cross(&v0, &v2);
                let c1 = cross(&v2, &v1);
                if c0 * c1 < 0.0 {
                    continue;
                }
                if dot(&v0, &v2) < 0.0 || dot(&v1, &v2) < 0.0 {
                    continue;
                }

                let pp = if c0 > 0.0 {
                    vec![
                        current_saddle.p,
                        refined[idx_j].p,
                        refined[idx_i].p,
                        refined[idx_k].p,
                    ]
                } else {
                    vec![
                        current_saddle.p,
                        refined[idx_k].p,
                        refined[idx_i].p,
                        refined[idx_j].p,
                    ]
                };
                let homo_points_option =
                    decode_positions(img.width(), img.height(), &pp, 2, 6, 0.5);
                if let Some(homo_points) = homo_points_option {
                    let bits = bit_code(&img_grey, &homo_points, 10, 5);
                    if bits.is_some() {
                        let tag_id_option = best_tag(bits.unwrap(), 3, &self.code_list, self.edge);
                        if tag_id_option.is_some() {
                            active_idxs.remove(&idx_i);
                            active_idxs.remove(&idx_j);
                            active_idxs.remove(&idx_k);
                            for next_idx in &closest_idxs {
                                if active_idxs.contains(next_idx) {
                                    start_idx = *next_idx;
                                    break;
                                }
                            }
                            avg_tag_l.push(avg_l);
                            let tag_id = tag_id_option.unwrap();

                            let mut pp = pp;
                            pp.rotate_left(tag_id.1);
                            let refined_arr: [(f32, f32); 4] = pp.try_into().unwrap();
                            detected_tags.insert(tag_id.0 as u32, refined_arr);
                            break;
                        }
                    }
                };
            }
        }
        detected_tags
    }
}
