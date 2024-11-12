use core::f32;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    ops::BitXor,
};

use crate::image_util::GrayImagef32;
use crate::saddle::Saddle;
use crate::{image_util, math_util, tag_families};
use faer::solvers::SpSolverLstsq;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};

pub struct TagDetector {
    edge: u8,
    border: u8,
    hamming_distance: u8,
    code_list: Vec<u64>,
    detector_params: DetectorParams,
}

pub struct DetectorParams {
    pub tag_spacing_ratio: f32,
    pub min_saddle_angle: f32,
    pub max_saddle_angle: f32,
}

impl DetectorParams {
    pub fn default_params() -> DetectorParams {
        DetectorParams {
            tag_spacing_ratio: 0.3,
            min_saddle_angle: 30.0,
            max_saddle_angle: 60.0,
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
    let h = image_util::tag_homography(quad_pts, side_bits, margin);
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

fn init_saddle_clusters(h_mat: &GrayImagef32, threshold: f32) -> Vec<Vec<(u32, u32)>> {
    let mut tmp_h_mat = h_mat.clone();
    let mut clusters = Vec::new();
    for r in 1..h_mat.height() - 1 {
        for c in 1..h_mat.width() - 1 {
            let mut cluster = Vec::new();
            image_util::pixel_bfs(&mut tmp_h_mat, &mut cluster, c, r, threshold);
            if cluster.len() > 0 {
                clusters.push(cluster);
            }
        }
    }
    clusters
}

pub fn saddle_distance2(s0: &Saddle, s1: &Saddle) -> f32 {
    let x = s0.p.0 - s1.p.0;
    let y = s0.p.1 - s1.p.1;
    x * x + y * y
}

fn closest_n_idx(
    saddles: &[Saddle],
    self_idx: usize,
    active_idxs: &HashSet<usize>,
    num: usize,
    same_polarity: bool,
) -> Vec<usize> {
    let target = saddles[self_idx];
    // let polarity = (target.theta - target.theta2).abs() < 1.0;
    let mut sorted: Vec<_> = saddles
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if active_idxs.contains(&i) {
                if same_polarity && math_util::theta_distance_degree(s.theta, target.theta) < 5.0 {
                    return Some((i, s.clone()));
                } else if !same_polarity
                    && math_util::theta_distance_degree(s.theta, target.theta) > 80.0
                {
                    return Some((i, s.clone()));
                }
            }
            None
        })
        .collect();
    sorted.sort_by(|(_, a), (_, b)| {
        saddle_distance2(&target, a)
            .partial_cmp(&saddle_distance2(&target, b))
            .unwrap()
    });
    let out_len = sorted.len().min(num);
    sorted[0..out_len].iter().map(|(i, _)| *i).collect()
}

pub struct Tag {
    pub id: u32,
    pub p: [(f32, f32); 4],
}

pub fn rochade_refine<T>(
    image_input: &ImageBuffer<Luma<T>, Vec<T>>,
    initial_corners: &Vec<(f32, f32)>,
    half_size_patch: i32,
) -> Vec<Saddle>
where
    T: image::Primitive + Into<f32>,
{
    const PIXEL_MOVE_THRESHOLD: f32 = 1.0;
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
            let (x0, y0) = math_util::find_xy(2.0 * a1, a2, a4, a2, 2.0 * a3, a5);
            // move too much
            if x0.abs() > PIXEL_MOVE_THRESHOLD || y0.abs() > PIXEL_MOVE_THRESHOLD {
                continue;
            } else {
                // Alturki, Abdulrahman S., and John S. Loomis.
                // "A new X-Corner Detection for Camera Calibration Using Saddle Points."
                let c5 = (a1 + a3) / 2.0;
                let c4 = (a1 - a3) / 2.0;
                let c3 = a2 / 2.0;
                let k = (c4 * c4 + c3 * c3).sqrt();
                let phi = (-1.0 * c5 / k).acos() / 2.0 / PI * 180.0;

                let theta = c3.atan2(c4) / 2.0 / PI * 180.0;

                if c5.abs() >= k {
                    continue;
                }
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
pub fn is_posible_quad(
    current_saddle: &Saddle,
    cross_saddle: &Saddle,
    side_saddle0: &Saddle,
    side_saddle1: &Saddle,
) -> bool {
    if math_util::theta_distance_degree(current_saddle.theta, cross_saddle.theta) > 10.0 {
        return false;
    }
    if math_util::theta_distance_degree(side_saddle0.theta, side_saddle1.theta) > 10.0 {
        return false;
    }

    let l0 = saddle_distance2(&current_saddle, &side_saddle0).sqrt();
    let l1 = saddle_distance2(&current_saddle, &side_saddle1).sqrt();
    let l2 = saddle_distance2(&cross_saddle, &side_saddle0).sqrt();
    let l3 = saddle_distance2(&cross_saddle, &side_saddle1).sqrt();
    let avg_l = (l0 + l1 + l2 + l3) / 4.0;
    let l_ratio = 0.3;
    let min_l = avg_l * (1.0 - l_ratio);
    let max_l = avg_l * (1.0 + l_ratio);
    if l0 < min_l
        || l0 > max_l
        || l1 < min_l
        || l1 > max_l
        || l2 < min_l
        || l2 > max_l
        || l3 < min_l
        || l3 > max_l
    {
        return false;
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
    let c0 = math_util::cross(&v0, &v2);
    let c1 = math_util::cross(&v2, &v1);
    if c0 * c1 < 0.0 {
        return false;
    }
    if math_util::dot(&v0, &v2) < 0.0 || math_util::dot(&v1, &v2) < 0.0 {
        return false;
    }
    true
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
                hamming_distance: 1,
                code_list: tag_families::T25H7.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T25H9 => TagDetector {
                edge: 5,
                border: 2,
                hamming_distance: 1,
                code_list: tag_families::T25H9.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T36H11 => TagDetector {
                edge: 6,
                border: 2,
                hamming_distance: 1,
                code_list: tag_families::T36H11.to_vec(),
                detector_params,
            },
            tag_families::TagFamily::T36H11B1 => TagDetector {
                edge: 6,
                border: 1,
                hamming_distance: 1,
                code_list: tag_families::T36H11.to_vec(),
                detector_params,
            },
        }
    }

    pub fn refined_saddle_points(&self, img: &DynamicImage) -> Vec<Saddle> {
        let blur: GrayImagef32 = imageproc::filter::gaussian_blur_f32(&img.to_luma32f(), 1.5);
        let hessian_response_mat = image_util::hessian_response(&blur);
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
        let saddle_points = rochade_refine(&blur, &saddle_cluster_centers, 2);
        let smax = saddle_points.iter().fold(f32::MIN, |acc, s| acc.max(s.k)) / 10.0;
        let refined: Vec<Saddle> = saddle_points
            .iter()
            .filter_map(|s| {
                if s.k < smax
                    || s.phi < self.detector_params.min_saddle_angle
                    || s.phi > self.detector_params.max_saddle_angle
                {
                    None
                } else {
                    Some(s.to_owned())
                }
            })
            .collect();
        refined
    }

    fn find_one_target(
        &self,
        refined: &[Saddle],
        img_grey: &GrayImage,
    ) -> Option<(Tag, HashSet<usize>)> {
        let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();

        let mut start_idx = active_idxs.iter().next().unwrap().clone();
        while active_idxs.len() >= 4 {
            if !active_idxs.remove(&start_idx) {
                start_idx = active_idxs.iter().next().unwrap().clone();
                continue;
            }
            let current_saddle = refined[start_idx];
            let closest_idxs_same = closest_n_idx(&refined, start_idx, &active_idxs, 15, true);
            let closest_idxs_diff = closest_n_idx(&refined, start_idx, &active_idxs, 30, false);
            for idx_i in &closest_idxs_same {
                for jk in closest_idxs_diff.iter().combinations(2) {
                    let idx_i = *idx_i;
                    let idx_j = *jk[0];
                    let idx_k = *jk[1];
                    let cross_saddle = refined[idx_i];
                    let side_saddle0 = refined[idx_j];
                    let side_saddle1 = refined[idx_k];
                    if !is_posible_quad(
                        &current_saddle,
                        &cross_saddle,
                        &side_saddle0,
                        &side_saddle1,
                    ) {
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
                    let c0 = math_util::cross(&v0, &v2);
                    let c1 = math_util::cross(&v2, &v1);
                    if c0 * c1 < 0.0 {
                        continue;
                    }
                    if math_util::dot(&v0, &v2) < 0.0 || math_util::dot(&v1, &v2) < 0.0 {
                        continue;
                    }

                    let quad_points = if c0 > 0.0 {
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
                    if let Some((tag_id, refined_arr)) =
                        self.try_decode_quad(&img_grey, &quad_points)
                    {
                        active_idxs.remove(&idx_i);
                        active_idxs.remove(&idx_j);
                        active_idxs.remove(&idx_k);
                        let tag = Tag {
                            id: tag_id as u32,
                            p: refined_arr,
                        };
                        return Some((tag, active_idxs));
                    }
                }
            }
        }
        None
    }

    fn try_decode_quad(
        &self,
        img_grey: &GrayImage,
        quad_points: &[(f32, f32)],
    ) -> Option<(usize, [(f32, f32); 4])> {
        let homo_points_option = decode_positions(
            img_grey.width(),
            img_grey.height(),
            quad_points,
            self.border,
            self.edge,
            0.5,
        );
        if let Some(homo_points) = homo_points_option {
            let bits_option = bit_code(&img_grey, &homo_points, 10, 3);
            if let Some(bits) = bits_option {
                let tag_id_option =
                    best_tag(bits, self.hamming_distance, &self.code_list, self.edge);
                if let Some((tag_id, rotation)) = tag_id_option {
                    let mut new_q_pts = quad_points.to_owned();
                    new_q_pts.rotate_left(rotation);
                    new_q_pts.reverse();
                    let refined_arr: [(f32, f32); 4] = new_q_pts.try_into().unwrap();
                    return Some((tag_id, refined_arr));
                }
            }
        }
        None
    }

    // TODO too slow
    pub fn detect(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let mut detected_tags = HashMap::new();
        let mut avg_tag_l = Vec::new();
        let img_grey = img.to_luma8();
        let refined = self.refined_saddle_points(&img);
        if refined.len() < 4 {
            return detected_tags;
        }
        let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();

        let mut start_idx = refined.len() / 2;
        while active_idxs.len() >= 4 {
            if !active_idxs.remove(&start_idx) {
                start_idx = active_idxs.iter().next().unwrap().clone();
                continue;
            }
            let current_saddle = refined[start_idx];
            // println!("start idx: {} {} {}", start_idx, refined[start_idx].p.0, refined[start_idx].p.1);
            let closest_idxs_same = closest_n_idx(&refined, start_idx, &active_idxs, 15, true);
            let closest_idxs_diff = closest_n_idx(&refined, start_idx, &active_idxs, 30, false);
            let mut found = false;
            for idx_i in &closest_idxs_same {
                if found {
                    break;
                }
                for jk in closest_idxs_diff.iter().combinations(2) {
                    let idx_i = *idx_i;
                    let idx_j = *jk[0];
                    let idx_k = *jk[1];
                    let cross_saddle = refined[idx_i];
                    let side_saddle0 = refined[idx_j];
                    let side_saddle1 = refined[idx_k];
                    if !is_posible_quad(
                        &current_saddle,
                        &cross_saddle,
                        &side_saddle0,
                        &side_saddle1,
                    ) {
                        continue;
                    }

                    let l0 = saddle_distance2(&current_saddle, &side_saddle0).sqrt();
                    let l1 = saddle_distance2(&current_saddle, &side_saddle1).sqrt();
                    let l2 = saddle_distance2(&cross_saddle, &side_saddle0).sqrt();
                    let l3 = saddle_distance2(&cross_saddle, &side_saddle1).sqrt();
                    let avg_l = (l0 + l1 + l2 + l3) / 4.0;
                    if avg_tag_l.len() > 4 {
                        let global_avg_l = avg_tag_l.iter().sum::<f32>() / avg_tag_l.len() as f32;
                        if avg_l < global_avg_l * 0.7 || avg_l > global_avg_l * 1.3 {
                            continue;
                        }
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
                    let c0 = math_util::cross(&v0, &v2);
                    let c1 = math_util::cross(&v2, &v1);
                    if c0 * c1 < 0.0 {
                        continue;
                    }
                    if math_util::dot(&v0, &v2) < 0.0 || math_util::dot(&v1, &v2) < 0.0 {
                        continue;
                    }

                    let quad_points = if c0 > 0.0 {
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
                    if let Some((tag_id, refined_arr)) =
                        self.try_decode_quad(&img_grey, &quad_points)
                    {
                        active_idxs.remove(&idx_i);
                        active_idxs.remove(&idx_j);
                        active_idxs.remove(&idx_k);
                        for next_idx in &closest_idxs_same {
                            if active_idxs.contains(next_idx) {
                                start_idx = *next_idx;
                                break;
                            }
                        }
                        avg_tag_l.push(avg_l);
                        detected_tags.insert(tag_id as u32, refined_arr);
                        found = true;
                        break;
                    }
                }
            }
        }
        detected_tags
    }

    pub fn detect2(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let mut detected_tags = HashMap::new();
        let img_grey = img.to_luma8();
        let refined = self.refined_saddle_points(&img);
        if refined.len() < 4 {
            return detected_tags;
        }
        if let Some((tag, _)) = self.find_one_target(&refined, &img_grey) {
            detected_tags.insert(tag.id, tag.p);
        }
        detected_tags
    }
    pub fn detect3(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let mut detected_tags = HashMap::new();
        let img_grey = img.to_luma8();
        let refined = self.refined_saddle_points(&img);
        let best_board_indexes_option = try_find_best_board(&refined);
        if let Some(best_board_indexes) = best_board_indexes_option {
            for quad_indexes in best_board_indexes {
                let quad_points: Vec<(f32, f32)> =
                    quad_indexes.iter().map(|i| refined[*i].p).collect();
                if let Some((tag_id, refined_arr)) = self.try_decode_quad(&img_grey, &quad_points) {
                    // let tag = Tag {
                    //     id: tag_id as u32,
                    //     p: refined_arr,
                    // };
                    detected_tags.insert(tag_id as u32, refined_arr);
                }
            }
        }
        detected_tags
    }
}

pub fn init_quads(refined: &[Saddle], s0_idx: usize, tree: &KdTree<f32, 2>) -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    let s0 = refined[s0_idx];
    let nearest = tree.nearest_n::<SquaredEuclidean>(&s0.arr(), 50);
    let mut same_p_idxs = Vec::new();
    let mut diff_p_idxs = Vec::new();
    for n in nearest {
        let s = refined[n.item as usize];
        let theta_diff = crate::math_util::theta_distance_degree(s0.theta, s.theta);
        if theta_diff < 3.0 {
            same_p_idxs.push(n.item as usize);
        } else if theta_diff > 80.0 {
            diff_p_idxs.push(n.item as usize);
        }
    }
    for s1_idx in same_p_idxs {
        let s1 = refined[s1_idx];
        for dp in diff_p_idxs.iter().combinations(2) {
            let d0 = refined[*dp[0]];
            let d1 = refined[*dp[1]];
            if !crate::saddle::is_valid_quad(&s0, &d0, &s1, &d1) {
                // if s1_idx == 30 && *dp[1] == 28 && *dp[0] == 60{
                //     panic!("aaaa");
                // }
                continue;
            }
            let v01 = (d0.p.0 - s0.p.0, d0.p.1 - s0.p.1);
            let v02 = (s1.p.0 - s0.p.0, s1.p.1 - s0.p.1);
            let c0 = crate::math_util::cross(&v01, &v02);
            let quad_idxs = if c0 > 0.0 {
                [s0_idx, *dp[0], s1_idx, *dp[1]]
            } else {
                [s0_idx, *dp[1], s1_idx, *dp[0]]
            };
            out.push(quad_idxs);
        }
    }
    out
}

pub fn try_find_best_board(refined: &[Saddle]) -> Option<Vec<[usize; 4]>> {
    let entries: Vec<[f32; 2]> = refined.iter().map(|r| r.p.try_into().unwrap()).collect();
    // use the kiddo::KdTree type to get up and running quickly with default settings
    let mut tree: KdTree<f32, 2> = (&entries).into();

    // quad search
    let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();
    let (mut best_score, mut best_board_option) = (0, None);
    let mut count = 0;
    while active_idxs.len() > 4 && count < 30 {
        // let mut tree = tree.clone();
        let s0_idx = active_idxs.iter().next().unwrap().clone();
        active_idxs.remove(&s0_idx);
        tree.remove(&refined[s0_idx].arr(), s0_idx as u64);
        let quads = init_quads(&refined, s0_idx, &tree);
        for q in quads {
            let board = crate::board::Board::new(&refined, &active_idxs, &q, 0.3, &tree);
            if board.score > best_score {
                best_score = board.score;
                best_board_option = Some(board);
            }
        }
        // TODO review this
        if best_score > 5 {
            active_idxs.insert(s0_idx.clone());
            tree.add(&refined[s0_idx].arr(), s0_idx as u64);
        }
        if best_score >= 36 {
            break;
        }
        count += 1;
    }
    if let Some(best_board) = best_board_option {
        let tag_idxs: Vec<[usize; 4]> = best_board.all_tag_indexes();
        Some(tag_idxs)
    } else {
        None
    }
}
