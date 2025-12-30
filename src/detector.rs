use core::f32;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    ops::BitXor,
};

use crate::image_util::GrayImagef32;
use crate::saddle::Saddle;
use crate::{image_util, math_util, tag_families};
use faer::linalg::solvers::SolveLstsqCore;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use itertools::Itertools;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;

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
    pub max_num_of_boards: u8,
}

impl DetectorParams {
    pub fn default_params() -> DetectorParams {
        DetectorParams {
            tag_spacing_ratio: 0.3,
            min_saddle_angle: 30.0,
            max_saddle_angle: 60.0,
            max_num_of_boards: 2,
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
    let affine_mat = image_util::tag_affine(quad_pts, side_bits, margin);
    Some(
        (border_bits..border_bits + edge_bits)
            .flat_map(|x| {
                (border_bits..border_bits + edge_bits)
                    .map(|y| {
                        let tp = faer::mat![[x as f32], [y as f32], [1.0]];
                        let tt = affine_mat.clone() * tp;
                        (tt[(0, 0)], tt[(1, 0)])
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
            let invalid_count =
                if (mid_b as i32 - *b as i32).abs() < valid_brightness_threshold as i32 {
                    invalid_count + 1
                } else {
                    invalid_count
                };
            if *b > mid_b {
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

const fn rotate_bits(bits: u64, edge_bits: u8) -> u64 {
    let edge_bits = edge_bits as usize;
    let mut b = 0u64;
    let mut count = 0;
    let mut r = (edge_bits - 1) as i32;
    while r >= 0 {
        let mut c = 0;
        while c < edge_bits {
            let idx = r as usize + c * edge_bits;
            b |= ((bits >> idx) & 1) << count;
            count += 1;
            c += 1;
        }
        r -= 1;
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

fn init_saddle_clusters(mut h_mat: GrayImagef32, threshold: f32) -> Vec<Vec<(u32, u32)>> {
    let mut clusters = Vec::new();
    let mut cluster = Vec::with_capacity(64); // Reuse buffer
    for r in 1..h_mat.height() - 1 {
        for c in 1..h_mat.width() - 1 {
            let v = unsafe { h_mat.unsafe_get_pixel(c, r).0[0] };
            if v < threshold {
                cluster.clear();
                image_util::pixel_bfs(&mut h_mat, &mut cluster, c, r, threshold);
                if !cluster.is_empty() {
                    clusters.push(cluster.clone());
                }
            }
        }
    }
    clusters
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

    let kernel_size = (half_size_patch * 2 + 1) as usize;
    let num_pixels = kernel_size * kernel_size;

    let p_mat = {
        let mut mat_a: faer::Mat<f32> = faer::Mat::ones(num_pixels, 6);
        let mut count = 0;
        for r in 0..kernel_size {
            for c in 0..kernel_size {
                let x = c as f32 - half_size_patch as f32;
                let y = r as f32 - half_size_patch as f32;
                mat_a[(count, 0)] = x * x;
                mat_a[(count, 1)] = x * y;
                mat_a[(count, 2)] = y * y;
                mat_a[(count, 3)] = x;
                mat_a[(count, 4)] = y;
                count += 1;
            }
        }

        let qr = mat_a.qr();
        let mut p_mat_tmp: faer::Mat<f32> = faer::Mat::zeros(6, num_pixels);
        let mut rhs: faer::Mat<f32> = faer::Mat::zeros(num_pixels, 1);
        for i in 0..num_pixels {
            rhs.fill(0.0);
            rhs[(i, 0)] = 1.0;
            qr.solve_lstsq_in_place_with_conj(faer::Conj::No, rhs.as_mut());
            for j in 0..6 {
                p_mat_tmp[(j, i)] = rhs[(j, 0)];
            }
        }
        // Now transpose it to num_pixels x 6 so that each column (fixed parameter j) is contiguous
        p_mat_tmp.transpose().to_owned()
    };

    // kernel computation
    let gamma = half_size_patch as f32;
    let flat_k: Vec<f32> = (0..kernel_size)
        .flat_map(|i| {
            (0..kernel_size).map(move |j| {
                0.0_f32.max(
                    gamma + 1.0
                        - ((gamma - i as f32) * (gamma - i as f32)
                            + (gamma - j as f32) * (gamma - j as f32))
                            .sqrt(),
                )
            })
        })
        .collect();
    let s = flat_k.iter().sum::<f32>();
    let flat_k: Vec<f32> = flat_k.iter().map(|v| v / s).collect();

    let (width, height) = (image_input.width() as i32, image_input.height() as i32);
    let half_size_patch2 = half_size_patch * 2;
    let img_raw = image_input.as_raw();
    let img_ptr = img_raw.as_ptr();
    let stride = width as usize;

    let mut smooth_sub_image = vec![0.0f32; num_pixels];

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

        let start_x = (round_x - half_size_patch2) as usize;
        let start_y = (round_y - half_size_patch2) as usize;

        // Convolution with raw pointers - Unrolled for kernel_size=5 (common case)
        if kernel_size == 5 {
            unsafe {
                let k = &flat_k;
                for r in 0..5 {
                    for c in 0..5 {
                        let mut conv_p = 0.0;
                        for pr in 0..5 {
                            let offset = (start_y + r + pr) * stride + start_x + c;
                            let rp = img_ptr.add(offset);
                            conv_p += (*rp).into() * k[pr * 5];
                            conv_p += (*rp.add(1)).into() * k[pr * 5 + 1];
                            conv_p += (*rp.add(2)).into() * k[pr * 5 + 2];
                            conv_p += (*rp.add(3)).into() * k[pr * 5 + 3];
                            conv_p += (*rp.add(4)).into() * k[pr * 5 + 4];
                        }
                        smooth_sub_image[r * 5 + c] = conv_p;
                    }
                }
            }
        } else {
            for r in 0..kernel_size {
                for c in 0..kernel_size {
                    let mut conv_p = 0.0;
                    let mut k_idx = 0;
                    for pr in 0..kernel_size {
                        unsafe {
                            let offset = (start_y + r + pr) * stride + start_x + c;
                            let row_ptr = img_ptr.add(offset);
                            for pc in 0..kernel_size {
                                let pixel_val: f32 = (*row_ptr.add(pc)).into();
                                conv_p += pixel_val * flat_k[k_idx];
                                k_idx += 1;
                            }
                        }
                    }
                    smooth_sub_image[r * kernel_size + c] = conv_p;
                }
            }
        }

        // Parameter estimation using scalar raw pointers
        let mut params = [0.0f32; 6];
        for (col_j, param) in p_mat.col_iter().zip(params.iter_mut()) {
            let mut sum = 0.0;
            for i in 0..num_pixels {
                sum += col_j[i] * smooth_sub_image[i];
            }
            *param = sum;
        }

        let a1 = params[0];
        let a2 = params[1];
        let a3 = params[2];
        let a4 = params[3];
        let a5 = params[4];
        let fxx = 2.0 * a1;
        let fyy = 2.0 * a3;
        let fxy = a2;
        let d = fxx * fyy - fxy * fxy;

        if d < 0.0 {
            let (x0, y0) = math_util::find_xy(2.0 * a1, a2, a4, a2, 2.0 * a3, a5);
            if x0.abs() <= PIXEL_MOVE_THRESHOLD && y0.abs() <= PIXEL_MOVE_THRESHOLD {
                let c5 = (a1 + a3) / 2.0;
                let c4 = (a1 - a3) / 2.0;
                let c3 = a2 / 2.0;
                let k = (c4 * c4 + c3 * c3).sqrt();
                if c5.abs() < k {
                    let phi = (-c5 / k).acos() / 2.0 / PI * 180.0;
                    let theta = c3.atan2(c4) / 2.0 / PI * 180.0;
                    refined_corners.push(Saddle {
                        p: (initial_x.round() + x0, initial_y.round() + y0),
                        k,
                        theta,
                        phi,
                    });
                }
            }
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
                hamming_distance: 2,
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

    pub fn refined_saddle_points(&self, img: &DynamicImage) -> Vec<Saddle> {
        let luma_f32 = img.to_luma32f();
        let blur: GrayImagef32 = image_util::gaussian_blur_f32(&luma_f32, 1.5);
        let hessian_response_mat = image_util::hessian_response(&blur);

        // Use references to find max/min to avoid allocations
        let min_response = hessian_response_mat
            .as_raw()
            .iter()
            .fold(f32::MAX, |acc, &e| acc.min(e));
        let min_response_threshold = min_response * 0.05;

        let saddle_clusters = init_saddle_clusters(hessian_response_mat, min_response_threshold);
        let saddle_cluster_centers: Vec<(f32, f32)> = saddle_clusters
            .iter()
            .map(|c| {
                let (sx, sy) = c.iter().fold((0.0, 0.0), |(ax, ay), &(ex, ey)| {
                    (ax + ex as f32, ay + ey as f32)
                });
                (sx / c.len() as f32, sy / c.len() as f32)
            })
            .collect();
        let saddle_points = rochade_refine(&blur, &saddle_cluster_centers, 2);

        if saddle_points.is_empty() {
            return Vec::new();
        }

        let s_max_k = saddle_points.iter().fold(f32::MIN, |acc, s| acc.max(s.k)) / 10.0;
        let refined: Vec<Saddle> = saddle_points
            .into_iter()
            .filter(|s| {
                s.k >= s_max_k
                    && s.phi >= self.detector_params.min_saddle_angle
                    && s.phi <= self.detector_params.max_saddle_angle
            })
            .collect();
        refined
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
            let bits_option = bit_code(img_grey, &homo_points, 10, 3);
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

    #[cfg(feature = "kornia")]
    pub fn detect_kornia<const N: usize>(
        &self,
        img: &kornia::image::Image<u8, N>,
    ) -> HashMap<u32, [(f32, f32); 4]> {
        let dyn_img = match img.num_channels() {
            1 => DynamicImage::ImageLuma8(
                GrayImage::from_vec(
                    img.width() as u32,
                    img.height() as u32,
                    img.clone().0.into_vec(),
                )
                .unwrap(),
            ),
            3 => DynamicImage::ImageRgb8(
                image::RgbImage::from_vec(
                    img.width() as u32,
                    img.height() as u32,
                    img.clone().0.into_vec(),
                )
                .unwrap(),
            ),
            _ => panic!("Only support u8c1 and u8c3"),
        };
        self.detect(&dyn_img)
    }

    pub fn detect(&self, img: &DynamicImage) -> HashMap<u32, [(f32, f32); 4]> {
        let mut detected_tags = HashMap::new();
        let img_grey = img.to_luma8();
        let mut refined = self.refined_saddle_points(img);

        for _ in 0..self.detector_params.max_num_of_boards {
            let best_board_indexes_option = try_find_best_board(&refined);
            if let Some(best_board_indexes) = best_board_indexes_option {
                let mut indexs_to_remove = HashSet::new();
                for quad_indexes in best_board_indexes {
                    let quad_points: Vec<(f32, f32)> =
                        quad_indexes.iter().map(|i| refined[*i].p).collect();
                    if let Some((tag_id, refined_arr)) =
                        self.try_decode_quad(&img_grey, &quad_points)
                    {
                        detected_tags.insert(tag_id as u32, refined_arr);
                        for qi in quad_indexes {
                            indexs_to_remove.insert(qi);
                        }
                    }
                }
                refined = refined
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| {
                        if indexs_to_remove.contains(&i) {
                            None
                        } else {
                            Some(*s)
                        }
                    })
                    .collect();
            }
        }
        detected_tags
    }
}

pub fn init_quads(
    refined: &[Saddle],
    s0_idx: usize,
    tree: &KdTree<f32, usize, [f32; 2]>,
) -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    let s0 = refined[s0_idx];
    let nearest = tree.nearest(&s0.arr(), 50, &squared_euclidean).unwrap();
    let mut same_p_idxs = Vec::new();
    let mut diff_p_idxs = Vec::new();
    for n in &nearest[1..] {
        let s_idx = *n.1;
        let s = refined[s_idx];
        let theta_diff = crate::math_util::theta_distance_degree(s0.theta, s.theta);
        if theta_diff < 5.0 {
            same_p_idxs.push(s_idx);
        } else if theta_diff > 80.0 {
            diff_p_idxs.push(s_idx);
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
    if refined.is_empty() {
        return None;
    }
    let mut tree = KdTree::new(2);
    for (i, r) in refined.iter().enumerate() {
        tree.add(r.arr(), i).unwrap();
    }

    // quad search
    let active_mask: Vec<bool> = vec![true; refined.len()];
    let (mut best_score, mut best_board_option) = (0, None);
    let mut count = 0;
    let mut hm = HashMap::<i32, Vec<usize>>::new();
    refined.iter().enumerate().for_each(|(i, s)| {
        let angle = s.theta.round() as i32;
        if let std::collections::hash_map::Entry::Vacant(e) = hm.entry(angle) {
            e.insert(vec![i]);
        } else {
            hm.get_mut(&angle).unwrap().push(i);
        }
    });
    let mut s0_idxs: Vec<usize> = hm
        .iter()
        .sorted_by(|a, b| a.1.len().cmp(&b.1.len()))
        .next_back()
        .unwrap()
        .1
        .to_owned();
    while !s0_idxs.is_empty() && count < 30 {
        let s0_idx = s0_idxs.pop().unwrap();
        let quads = init_quads(refined, s0_idx, &tree);
        for q in quads {
            let board = crate::board::Board::new(refined, &active_mask, &q, 0.3, &tree);
            if board.score > best_score {
                best_score = board.score;
                best_board_option = Some(board);
            }
        }
        if best_score >= 36 {
            break;
        }
        count += 1;
    }
    if let Some(mut best_board) = best_board_option {
        best_board.try_fix_missing();
        let tag_idxs: Vec<[usize; 4]> = best_board.all_tag_indexes();
        Some(tag_idxs)
    } else {
        None
    }
}
