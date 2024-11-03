use aprilgrid_rs::quad;
use glob::glob;
use image::{
    imageops::FilterType::Nearest, DynamicImage, GenericImageView, GrayImage, ImageReader,
};
use imageproc::{contours::find_contours, morphology::dilate};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::{io::Cursor, ops::BitXor};

fn log_image_as_compressed(recording: &RecordingStream, topic: &str, img: &DynamicImage) {
    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();

    recording
        .log(
            format!("{}/image", topic),
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

fn decode_positions(
    img: &GrayImage,
    quad_pts: &[(f32, f32)],
    border_bits: u8,
    edge_bits: u8,
) -> Option<Vec<(f32, f32)>> {
    if quad_pts.iter().any(|(x, y)| {
        let x = x.round() as u32;
        let y = y.round() as u32;
        x >= img.width() || y >= img.height()
    }) {
        return None;
    }
    let side_bits = border_bits * 2 + edge_bits;
    let h = aprilgrid_rs::homography::tag_homography(quad_pts, side_bits);
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

fn bit_code(img: &GrayImage, decode_pts: &[(f32, f32)]) -> Option<u64> {
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
            if (mid_b as i32 - *b as i32).abs() < 30 {
                (acc, invalid_count + 1)
            } else if *b > mid_b {
                (acc | (1 << i), invalid_count)
            } else {
                (acc, invalid_count)
            }
        },
    );
    if invalid_count > 5 {
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

fn best_tag(bits: u64, thres: u8) -> Option<(usize, usize)> {
    let mut bits = bits;
    for rotated in 0..4 {
        let scores: Vec<u32> = aprilgrid_rs::tag_families::T36H11
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
        bits = rotate_bits(bits, 6);
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    let dataset_root = "data0";
    // let dataset_root =
    //     "/Users/powei/Documents/dataset/tum_vi/dataset-calib-cam1_1024_16/mav0/cam0/data";
    let img_paths = glob(format!("{}/*.png", dataset_root).as_str()).expect("failed");
    let mut time_sec = 0.0;
    let fps = 30.0;
    let one_frame_time = 1.0 / fps;
    for path in img_paths {
        let time_ns: i64 = path
            .as_ref()
            .unwrap()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let img0 = ImageReader::open(path.unwrap())?.decode()?;

        let img0_grey = img0.to_luma8();
        let quads = aprilgrid_rs::quad::find_quad(&img0, 400.0);

        let mut valid_tag = Vec::new();
        let mut colors = Vec::new();
        // recording.set_time_nanos("stable_time", time_ns);
        recording.set_time_seconds("stable_time", time_sec);
        time_sec += one_frame_time;

        // // debug
        // let img0_bri = aprilgrid_rs::quad::adjust_brightness(&img0, 100);
        // let img0_grey_contrast = img0_bri.adjust_contrast(500.0);
        // let thredhold_image = dilate(
        //     &img0_grey_contrast.to_luma8(),
        //     imageproc::distance_transform::Norm::LInf,
        //     2,
        // );
        // log_image_as_compressed(&recording, "/cam0_bri", &img0_bri);
        // log_image_as_compressed(&recording, "/cam0_con", &img0_grey_contrast);
        // log_image_as_compressed(&recording, "/cam0_thres", &DynamicImage::ImageLuma8(thredhold_image));
        // let contours = find_contours::<u32>(&max_pool);

        for (i, c) in quads.iter().enumerate() {
            println!("{}", i);
            // recording
            //     .log(
            //         format!("/cam0/quad{}", i),
            //         &rerun::Points2D::new(rerun_shift(c))
            //             .with_radii([rerun::Radius::new_ui_points(2.0)]),
            //     )
            //     .expect("msg");
            let homo_points_option = decode_positions(&img0_grey, c, 2, 6);
            if let Some(mut homo_points) = homo_points_option {
                let bits = bit_code(&img0_grey, &homo_points);
                if bits.is_none() {
                    continue;
                }
                let tag_id_option = best_tag(bits.unwrap(), 4);
                if tag_id_option.is_none() {
                    continue;
                }
                let tag_id = tag_id_option.unwrap();
                colors.append(&mut vec![id_to_color(tag_id.0); homo_points.len()]);
                valid_tag.append(&mut homo_points);

                // let mut intersect_points = c.clone();
                // intersect_points.rotate_left(tag_id.1);
                // intersect_points.push(intersect_points[0]);
                // let li = rerun::LineStrip2D::from_iter(rerun_shift(&intersect_points));
                // recording
                //     .log(
                //         "/cam0/intersec_pts".to_string(),
                //         &rerun::LineStrips2D::new([li])
                //             .with_colors(vec![id_to_color(i)])
                //             .with_radii([rerun::Radius::new_ui_points(1.0)]),
                //     )
                //     .expect("msg");
            }
        }
        log_image_as_compressed(&recording, "/cam0", &img0);
        recording
            .log(
                format!("/cam0/tag_position"),
                &rerun::Points2D::new(rerun_shift(&valid_tag))
                    .with_colors(colors)
                    .with_radii([rerun::Radius::new_ui_points(2.0)]),
            )
            .expect("msg");
    }

    Ok(())
}
