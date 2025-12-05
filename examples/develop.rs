use aprilgrid::board::Board;
use aprilgrid::detector::{best_tag, bit_code, decode_positions, try_find_best_board};
use aprilgrid::saddle::{Saddle, is_valid_quad};
use core::f32;
use glam::{Vec2, Vec2Swizzles};
use glob::glob;
use image::{
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageReader,
    imageops::FilterType::{Nearest, Triangle},
};
use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::fmt::format;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    io::Cursor,
    usize,
};

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
            &rerun::EncodedImage::from_file_contents(bytes),
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
            &rerun::EncodedImage::from_file_contents(bytes),
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    let args = Args::parse();
    let img_path = args.img;
    let img = ImageReader::open(img_path)?.decode()?;
    let detector = aprilgrid::detector::TagDetector::new(&aprilgrid::TagFamily::T36H11, None);
    let refined = detector.refined_saddle_points(&img);
    let img_grey = img.to_luma8();

    log_image_as_compressed(&recording, "/cam0", &img);
    // let best_board = try_find_best_board(&refined);
    // if let Some(board) = best_board {
    //     let mut pts = Vec::new();
    //     let mut colors = Vec::new();
    //     board.iter().for_each(|q| {
    //         for i in q {
    //             pts.push(refined[*i].p);
    //         }
    //         colors.push((255, 0, 0, 255));
    //         colors.push((255, 255, 0, 255));
    //         colors.push((255, 0, 255, 255));
    //         colors.push((0, 255, 255, 255));
    //     });
    //     recording
    //         .log(
    //             "/cam0/image/board",
    //             &rerun::Points2D::new(rerun_shift(&pts))
    //                 .with_colors(colors)
    //                 .with_labels([format!("{}", board.len())]),
    //         )
    //         .unwrap();
    // }

    let entries: Vec<[f32; 2]> = refined.iter().map(|r| r.p.try_into().unwrap()).collect();
    // use the kiddo::KdTree type to get up and running quickly with default settings
    let tree: KdTree<f32, 2> = (&entries).into();

    // quad search
    let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();
    let mut count = 0;

    while active_idxs.len() > 4 {
        let mut tree = tree.clone();
        let s0_idx = active_idxs.iter().next().unwrap().clone();
        println!("s0 {}", s0_idx);
        // let s0_idx: usize = 86;
        active_idxs.remove(&s0_idx);
        tree.remove(&refined[s0_idx].arr(), s0_idx as u64);
        let quads = aprilgrid::detector::init_quads(&refined, s0_idx, &tree);
        for q in quads {
            let points = vec![
                refined[q[0]].p,
                refined[q[1]].p,
                refined[q[2]].p,
                refined[q[3]].p,
                refined[q[0]].p,
            ];
            let board = Board::new(&refined, &active_idxs, &q, 0.3, &tree);
            let mut pts = Vec::new();
            let mut colors = Vec::new();
            board.all_tag_indexes().iter().for_each(|q| {
                for i in q {
                    pts.push(refined[*i].p);
                }
                colors.push((255, 0, 0, 255));
                colors.push((255, 255, 0, 255));
                colors.push((255, 0, 255, 255));
                colors.push((0, 255, 255, 255));
            });
            recording
                .log(
                    "/cam0/image/board",
                    &rerun::Points2D::new(rerun_shift(&pts)).with_colors(colors),
                )
                .unwrap();
            recording.log(
                "quad",
                &rerun::LineStrips2D::new([rerun_shift(&points)])
                    .with_labels([format!("board score {}", board.score)]),
            )?;
        }
        break;
    }

    let cs: Vec<(f32, f32)> = refined.iter().map(|s| s.p).collect();
    let logs: Vec<String> = refined.iter().map(|s| format!("{:?}", s)).collect();

    println!("after {}", cs.len());
    recording
        .log(
            format!("/cam0/image/cluster_c"),
            &rerun::Points2D::new(rerun_shift(&cs))
                .with_radii([rerun::Radius::new_ui_points(3.0)])
                .with_labels(logs),
        )
        .expect("msg");

    Ok(())
}
