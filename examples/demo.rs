use glob::glob;
use image::{DynamicImage, ImageReader};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::io::Cursor;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    let dataset_root = "data";
    // let dataset_root = "/Users/powei/Documents/dataset/EuRoC/calibration/mav0/cam0/data";
    // let dataset_root =
    //     "/Users/powei/Documents/dataset/tum_vi/dataset-calib-cam1_1024_16/mav0/cam0/data";
    // let dataset_root = "test_data/data3";
    let img_paths = glob(format!("{}/*.png", dataset_root).as_str()).expect("failed");
    // let mut time_sec = 0.0;
    // let fps = 60.0;
    // let one_frame_time = 1.0 / fps;
    // let detector_params = None;
    let detector = aprilgrid::detector::TagDetector::new(&aprilgrid::TagFamily::T36H11, None);
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

        let mut corner_colors = Vec::new();
        let mut corner_list = Vec::new();
        let mut decode_points = Vec::new();
        let mut tag_colors = Vec::new();
        let mut memo = Vec::new();

        recording.set_time_nanos("stable_time", time_ns);
        // recording.set_time_seconds("stable_time", time_sec);
        // time_sec += one_frame_time;
        let tags = detector.detect(&img0);
        for (t_id, corners) in tags {
            let mut c: Vec<(f32, f32)> = corners.into();
            if let Some(mut homography_points) =
                aprilgrid::detector::decode_positions(img0.width(), img0.height(), &c, 2, 6, 0.5)
            {
                tag_colors.append(&mut vec![
                    id_to_color(t_id as usize);
                    homography_points.len()
                ]);
                decode_points.append(&mut homography_points);
            }
            corner_colors.push((255, 0, 0, 255));
            corner_colors.push((255, 255, 0, 255));
            corner_colors.push((255, 0, 255, 255));
            corner_colors.push((0, 255, 255, 255));
            corner_list.append(&mut c);
            for i in 0..4 {
                memo.push(format!("t{} {}", t_id, i));
            }
        }

        log_image_as_compressed(&recording, "/cam0", &img0);
        // log_image_as_compressed(&recording, "/cam0_contrast", &img0.adjust_contrast(200.0));
        // log_image_as_compressed(&recording, "/cam0_color", &img0);
        recording
            .log(
                format!("/cam0/corners"),
                &rerun::Points2D::new(rerun_shift(&corner_list))
                    .with_colors(corner_colors)
                    .with_radii([rerun::Radius::new_ui_points(2.0)]), // .with_labels(memo),
            )
            .expect("msg");
        recording
            .log(
                format!("/cam0/decode"),
                &rerun::Points2D::new(rerun_shift(&decode_points))
                    .with_colors(tag_colors)
                    .with_radii([rerun::Radius::new_ui_points(2.0)]),
            )
            .expect("msg");
    }

    Ok(())
}
