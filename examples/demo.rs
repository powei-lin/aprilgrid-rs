use aprilgrid_rs::quad;
use glob::glob;
use image::{DynamicImage, GrayImage, ImageReader};
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

fn bit_code(
    img: &GrayImage,
    quad_pts: &[(f32, f32)],
    border_bits: u8,
    edge_bits: u8,
) -> Vec<(f32, f32)> {
    if quad_pts.iter().any(|(x, y)| {
        let x = x.round() as u32;
        let y = y.round() as u32;
        x >= img.width() || y >= img.height()
    }) {
        return Vec::new();
    }
    let side_bits = border_bits * 2 + edge_bits;
    let h = aprilgrid_rs::homography::tag_homography(quad_pts, side_bits);
    let homo_points: Vec<(f32, f32)> = (border_bits..border_bits + edge_bits)
        .flat_map(|x| {
            (border_bits..border_bits + edge_bits)
                .map(|y| {
                    let tp = faer::mat![[x as f32], [y as f32], [1.0]];
                    let tt = h.clone() * tp;
                    (tt[(0, 0)] / tt[(2, 0)], tt[(1, 0)] / tt[(2, 0)])
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let brightness_vec: Vec<u8> = homo_points
        .iter()
        .map(|(x, y)| img.get_pixel(x.round() as u32, y.round() as u32).0[0])
        .collect();
    let (min_b, max_b) = brightness_vec
        .iter()
        .fold((255, 0), |(min_b, max_b), e| (min_b.min(*e), max_b.max(*e)));
    let avg_b = ((min_b as f32 + max_b as f32) / 2.0).round() as u8;
    let bits: u64 = brightness_vec
        .iter()
        .rev()
        .enumerate()
        .fold(
            0u64,
            |acc, (i, b)| {
                if *b > avg_b {
                    acc | (1 << i)
                } else {
                    acc
                }
            },
        );

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
    println!("best {} {}", best_idx, best_score);
    // println!("{:036b}", aprilgrid_rs::tag_families::T36H11[0]);
    // println!("{:036b}", bits);
    // let a = 12334;
    // a.con
    homo_points
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    let dataset_root = "data";
    let img_paths = glob(format!("{}/*.png", dataset_root).as_str()).expect("failed");

    for path in img_paths {
        let img0 = ImageReader::open(path.unwrap())?.decode()?;

        // let img0_grey = DynamicImage::ImageLuma8(img0_luma8);
        log_image_as_compressed(&recording, "/cam0", &img0);
        let quads = aprilgrid_rs::quad::find_quad(&img0, 100.0);
        for (i, c) in quads.iter().enumerate() {
            if i != 45 {
                continue;
            }
            println!("{}", i);
            let homo_points = bit_code(&img0.to_luma8(), c, 2, 6);
            recording
                .log(
                    "/cam0/tag_position".to_string(),
                    &rerun::Points2D::new(rerun_shift(&homo_points))
                        .with_radii([rerun::Radius::new_ui_points(3.0)]),
                )
                .expect("msg");
            let mut intersect_points = c.clone();
            intersect_points.push(intersect_points[0]);
            let li = rerun::LineStrip2D::from_iter(rerun_shift(&intersect_points));
            recording
                .log(
                    "/cam0/intersec_pts".to_string(),
                    &rerun::LineStrips2D::new([li])
                        .with_colors(vec![id_to_color(i)])
                        .with_radii([rerun::Radius::new_ui_points(1.0)]),
                )
                .expect("msg");
        }
    }

    Ok(())
}
