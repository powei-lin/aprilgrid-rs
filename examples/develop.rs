use aprilgrid::detector::{best_tag, bit_code, decode_positions, Saddle};
use core::f32;
use glob::glob;
use image::{
    imageops::FilterType::{Nearest, Triangle},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageReader,
};
use itertools::Itertools;
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

    // quad search
    let mut active_idxs: HashSet<usize> = (0..refined.len()).into_iter().collect();
    let mut count = 0;
    let mut start_idx = active_idxs.iter().next().unwrap().clone();

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
