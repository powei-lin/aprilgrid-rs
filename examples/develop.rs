use aprilgrid::detector::{best_tag, bit_code, decode_positions, Saddle};
use core::f32;
use glob::glob;
use image::{
    imageops::FilterType::{Nearest, Triangle},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageReader,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::{collections::HashSet, io::Cursor};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    img: String,
}

fn log_image_as_compressed(recording: &RecordingStream, topic: &str, img: &DynamicImage) {
    let mut bytes: Vec<u8> = Vec::new();
    img.to_luma8()
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();

    recording
        .log(
            format!("{}/image", topic),
            &rerun::Image::from_file_contents(bytes, None).unwrap(),
        )
        .unwrap();
}
fn log_grey_image_as_compressed(recording: &RecordingStream, topic: &str, img: &GrayImage) {
    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();

    recording
        .log(
            format!("{}/image_grey", topic),
            &rerun::Image::from_file_contents(bytes, None).unwrap(),
        )
        .unwrap();
}
fn id_to_color(id: usize) -> (u8, u8, u8, u8) {
    let mut rng = ChaCha8Rng::seed_from_u64(id as u64);
    let color_num = rng.gen_range(0..2u32.pow(24));

    (
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
        255,
    )
}

/// rerun use top left corner as (0, 0)
fn rerun_shift(p2ds: &[(f32, f32)]) -> Vec<(f32, f32)> {
    p2ds.iter().map(|(x, y)| (*x + 0.5, *y + 0.5)).collect()
}

type GrayImagef32 = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    // let mut time_sec = 0.0;
    // let fps = 60.0;
    // let one_frame_time = 1.0 / fps;
    // let detector_params = None;
    let args = Args::parse();
    let img_path = args.img;
    let img = ImageReader::open(img_path)?.decode()?;
    let detector = aprilgrid::detector::TagDetector::new(&aprilgrid::TagFamily::T36H11, None);
    let refined = detector.refined_saddle_points(&img);
    let img_grey = img.to_luma8();
    // let small = image::imageops::resize(&img.to_luma32f(), 100, 100, Triangle);
    // let blur: image::ImageBuffer<image::Luma<f32>, Vec<f32>> =
    //     imageproc::filter::gaussian_blur_f32(&img.to_luma32f(), 1.5);
    // let h = hessian_response(&blur);
    // // let mm = h.to_vec().iter().fold(f32::MIN, |acc, e|{acc.max(*e)});
    // let mn = h.to_vec().iter().fold(f32::MAX, |acc, e| acc.min(*e));
    // let init_saddle_points: Vec<_> = h
    //     .iter()
    //     .enumerate()
    //     .filter_map(|(i, p)| {
    //         if *p < mn * 0.05 {
    //             Some((i as u32 % h.width(), i as u32 / h.width()))
    //         } else {
    //             None
    //         }
    //     })
    //     .collect();
    // let f32pts: Vec<_> = init_saddle_points
    //     .iter()
    //     .map(|(x, y)| (*x as f32, *y as f32))
    //     .collect();
    // let saddle_clusters = init_saddle_clusters(&h, mn * 0.05);
    // let saddle_cluster_centers: Vec<(f32, f32)> = saddle_clusters
    //     .iter()
    //     .map(|c| {
    //         let (sx, sy) = c.iter().fold((0.0, 0.0), |(ax, ay), (ex, ey)| {
    //             (ax + *ex as f32, ay + *ey as f32)
    //         });
    //         (sx / c.len() as f32, sy / c.len() as f32)
    //     })
    //     .collect();

    log_image_as_compressed(&recording, "/cam0", &img);
    // quad search
    let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();
    let mut count = 0;
    let mut start_idx = active_idxs.iter().next().unwrap().clone();
    while active_idxs.len() >= 4 {
        if !active_idxs.remove(&start_idx) {
            start_idx = active_idxs.iter().next().unwrap().clone();
            continue;
        }
        let current_saddle = refined[start_idx];
        // println!("start idx: {} {} {}", start_idx, refined[start_idx].p.0, refined[start_idx].p.1);
        let closest_idxs = closest_n_idx(&refined, start_idx, &active_idxs, 50);

        let mut found = false;
        for i in 0..closest_idxs.len() {
            if found {
                break;
            }
            let idx_i = closest_idxs[i];
            let cross_saddle = refined[idx_i];
            if (current_saddle.theta - cross_saddle.theta).abs() > 10.0 {
                continue;
            }
            for j in 0..closest_idxs.len() {
                if found {
                    break;
                }
                if j == i {
                    continue;
                }
                let idx_j = closest_idxs[j];
                let side_saddle0 = refined[idx_j];
                for k in 0..closest_idxs.len() {
                    if found {
                        break;
                    }
                    if k == j || k == i {
                        continue;
                    }
                    let idx_k = closest_idxs[k];
                    let side_saddle1 = refined[idx_k];
                    if (side_saddle0.theta - side_saddle1.theta).abs() > 10.0 {
                        continue;
                    }
                    // let current_pp = vec![
                    //     current_saddle.p,
                    //     refined[idx_j].p,
                    //     refined[idx_i].p,
                    //     refined[idx_k].p,
                    // ];
                    // let color = vec![
                    //     (255, 0, 0, 255),
                    //     (0, 255, 0, 255),
                    //     (0, 0, 255, 255),
                    //     (255, 0, 255, 255),
                    // ];
                    // recording
                    //     .log(
                    //         format!("/cam0/image/current_p"),
                    //         &rerun::Points2D::new(rerun_shift(&current_pp))
                    //             .with_radii([rerun::Radius::new_ui_points(2.0)])
                    //             .with_colors(color),
                    //     )
                    //     .expect("msg");
                    // const theta_thres: f32 = 60.0;
                    // if (current_saddle.theta - side_saddle0.theta).abs() < theta_thres
                    //     || (current_saddle.theta - side_saddle1.theta).abs() < theta_thres
                    // {
                    //     recording.log("/log/small_theta", &rerun::TextLog::new("")).unwrap();
                    //     continue;
                    // }
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
                    let color = vec![
                        (255, 0, 0, 255),
                        (0, 255, 0, 255),
                        (0, 0, 255, 255),
                        (255, 0, 255, 255),
                    ];
                    recording
                        .log(
                            format!("/cam0/image/q"),
                            &rerun::Points2D::new(rerun_shift(&pp))
                                .with_radii([rerun::Radius::new_ui_points(2.0)])
                                .with_colors(color)
                                .with_labels([format!("q{}", count).as_str()]),
                        )
                        .expect("msg");
                    let homo_points_option =
                        decode_positions(img.width(), img.height(), &pp, 2, 6, 0.5);
                    if let Some(homo_points) = homo_points_option {
                        recording
                            .log(
                                format!("/cam0/image/h"),
                                &rerun::Points2D::new(rerun_shift(&homo_points))
                                    .with_radii([rerun::Radius::new_ui_points(2.0)]),
                            )
                            .expect("msg");
                        let bits = bit_code(&img_grey, &homo_points, 10, 5);
                        if bits.is_some() {
                            let tag_id_option = best_tag(
                                bits.unwrap(),
                                3,
                                &aprilgrid::tag_families::T36H11.to_vec(),
                                6,
                            );
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
                                found = true;
                                let tag_id = tag_id_option.unwrap();

                                // recording.set_time_nanos("stable_time", count*10000000);
                                let color = vec![
                                    (255, 0, 0, 255),
                                    (0, 255, 0, 255),
                                    (0, 0, 255, 255),
                                    (255, 0, 255, 255),
                                ];
                                recording
                                    .log(
                                        format!("/cam0/image/tag{}", tag_id.0),
                                        &rerun::Points2D::new(rerun_shift(&pp))
                                            .with_radii([rerun::Radius::new_ui_points(2.0)])
                                            // .with_colors(color)
                                            .with_labels([format!("t{}", tag_id.0).as_str()]),
                                    )
                                    .expect("msg");
                                println!("{}", count);
                                println!("s0 {:?}", current_saddle);
                                println!("s1 {:?}", side_saddle0);
                                println!("s2 {:?}", side_saddle1);
                                println!("s3 {:?}", cross_saddle);
                                println!();
                                count += 1;
                            }
                        }
                    };
                }
            }
        }
    }

    let cs: Vec<(f32, f32)> = refined.iter().map(|s| s.p).collect();
    let logs: Vec<String> = refined.iter().map(|s| format!("{:?}", s)).collect();

    println!("after {}", cs.len());
    // let m = imageproc::morphology::dilate(&m, imageproc::distance_transform::Norm::L1, 1);

    // let b = GrayImage::

    // let img_grey = imageproc::contrast::adaptive_threshold(&blur, 2);
    // let img_d = imageproc::morphology::dilate(&img_grey, imageproc::distance_transform::Norm::L1, 1);
    // let edge = imageproc::edges::canny(&blur, 10.0, 50.0);

    // recording.set_time_nanos("stable_time", time_ns);
    // recording.set_time_seconds("stable_time", time_sec);
    // time_sec += one_frame_time;

    // recording
    //     .log(
    //         format!("/cam0/image/cluster"),
    //         &rerun::Points2D::new(rerun_shift(&f32pts))
    //             .with_radii([rerun::Radius::new_ui_points(1.0)]),
    //     )
    //     .expect("msg");
    // recording
    //     .log(
    //         format!("/cam0/image/corners_center"),
    //         &rerun::Points2D::new(rerun_shift(&saddle_cluster_centers))
    //             .with_radii([rerun::Radius::new_ui_points(1.0)]),
    //     )
    //     .expect("msg");
    recording
        .log(
            format!("/cam0/image/cluster_c"),
            &rerun::Points2D::new(rerun_shift(&cs))
                .with_radii([rerun::Radius::new_ui_points(3.0)])
                .with_labels(logs),
        )
        .expect("msg");
    // log_grey_image_as_compressed(&recording, "/cam0", &img_d);
    // log_grey_image_as_compressed(&recording, "/edge", &edge);

    Ok(())
}
