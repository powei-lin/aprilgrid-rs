use aprilgrid::detector::{best_tag, bit_code, decode_positions, Saddle};
use core::f32;
use glob::glob;
use image::{
    imageops::FilterType::{Nearest, Triangle},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageReader,
};
use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::{
    collections::{HashMap, HashSet},
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

fn is_valid_quad(s0: &Saddle, d0: &Saddle, s1: &Saddle, d1: &Saddle) -> bool {
    if aprilgrid::math_util::theta_distance(d0.theta, d1.theta) > 3.0 {
        return false;
    }
    let v01 = (d0.p.0 - s0.p.0, d0.p.1 - s0.p.1);
    let v03 = (d1.p.0 - s0.p.0, d1.p.1 - s0.p.1);
    let v02 = (s1.p.0 - s0.p.0, s1.p.1 - s0.p.1);
    let c0 = aprilgrid::math_util::cross(&v01, &v02);
    let c1 = aprilgrid::math_util::cross(&v02, &v03);
    if c0 * c1 < 0.0 {
        return false;
    }
    let v12 = (s1.p.0 - d0.p.0, s1.p.1 - d0.p.1);
    let v23 = (d1.p.0 - s1.p.0, d1.p.1 - s1.p.1);
    let c01 = aprilgrid::math_util::cross(&v01, &v12);
    let c12 = aprilgrid::math_util::cross(&v12, &v23);
    if c01 * c12 < 0.0 {
        return false;
    }
    let v30 = (s0.p.0 - d1.p.0, s0.p.1 - d1.p.1);
    let a0 = aprilgrid::math_util::angle_degree(&v01, &v12);
    let a1 = aprilgrid::math_util::angle_degree(&v12, &v23);
    let a2 = aprilgrid::math_util::angle_degree(&v23, &v30);
    let a3 = aprilgrid::math_util::angle_degree(&v30, &v01);
    if (a0 - a2).abs() > 10.0 || (a1 - a3).abs() > 10.0 {
        return false;
    }
    if aprilgrid::math_util::dot(&v01, &v02) < 0.0 || aprilgrid::math_util::dot(&v03, &v02) < 0.0 {
        return false;
    }
    true
}

#[derive(Hash, PartialEq, Eq)]
struct BoardIdx {
    x: i32,
    y: i32,
}
impl BoardIdx {
    pub fn new(x: i32, y: i32) -> BoardIdx {
        BoardIdx { x, y }
    }
}

struct Board<'a> {
    refined: &'a [Saddle],
    active_idxs: HashSet<usize>,
    found_board_idxs: HashMap<BoardIdx, Option<[usize; 4]>>,
    tree: KdTree<f32, 2>,
    score: u32,
}
impl<'a> Board<'a> {
    pub fn new(
        refined: &'a [Saddle],
        active_idxs: &HashSet<usize>,
        quad_idxs: &[usize; 4],
        tree: &KdTree<f32, 2>,
    ) -> Board<'a> {
        let mut active_idxs = active_idxs.clone();
        let mut tree = tree.clone();
        for i in &quad_idxs[1..] {
            active_idxs.remove(i);
            tree.remove(&refined[*i].arr(), *i as u64);
        }
        Board {
            refined,
            active_idxs,
            found_board_idxs: HashMap::from([(BoardIdx::new(0, 0), Some(quad_idxs.clone()))]),
            tree,
            score: 1,
        }
    }
}

fn init_quads(refined: &[Saddle], s0_idx: usize, tree: &KdTree<f32, 2>) -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    let s0 = refined[s0_idx];
    let nearest = tree.nearest_n::<SquaredEuclidean>(&s0.arr(), 50);
    let mut same_p_idxs = Vec::new();
    let mut diff_p_idxs = Vec::new();
    for n in nearest {
        let s = refined[n.item as usize];
        let theta_diff = aprilgrid::math_util::theta_distance(s0.theta, s.theta);
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
            if !is_valid_quad(&s0, &d0, &s1, &d1) {
                continue;
            }
            let v01 = (d0.p.0 - s0.p.0, d0.p.1 - s0.p.1);
            let v02 = (s1.p.0 - s0.p.0, s1.p.1 - s0.p.1);
            let c0 = aprilgrid::math_util::cross(&v01, &v02);
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

    let entries: Vec<[f32; 2]> = refined.iter().map(|r| r.p.try_into().unwrap()).collect();
    // use the kiddo::KdTree type to get up and running quickly with default settings
    let tree: KdTree<f32, 2> = (&entries).into();

    // quad search
    let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();
    let mut count = 0;

    while active_idxs.len() > 4 {
        let mut tree = tree.clone();
        let s0_idx = active_idxs.iter().next().unwrap().clone();
        tree.remove(&refined[s0_idx].arr(), s0_idx as u64);
        let quads = init_quads(&refined, s0_idx, &tree);
        for q in quads {
            let points = [
                refined[q[0]].arr(),
                refined[q[1]].arr(),
                refined[q[2]].arr(),
                refined[q[3]].arr(),
                refined[q[0]].arr(),
            ];
            let board = Board::new(&refined, &active_idxs, &q, &tree);
            recording.log("quad", &rerun::LineStrips2D::new([points]))?;
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
