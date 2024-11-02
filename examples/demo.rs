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
    let img_paths = glob(format!("{}/*.png", dataset_root).as_str()).expect("failed");

    for path in img_paths {
        let img0 = ImageReader::open(path.unwrap())?.decode()?;

        // let img0_grey = DynamicImage::ImageLuma8(img0_luma8);
        log_image_as_compressed(&recording, "/cam0", &img0);
        let quads = aprilgrid_rs::quad::find_quad(&img0, 100.0);
        for (i, c) in quads.iter().enumerate() {
            let h = aprilgrid_rs::homography::tag_homography(c, 10);
            let homo_points: Vec<(f32, f32)> = (0..10)
                .flat_map(|x| {
                    (0..10)
                        .map(|y| {
                            let tp = faer::mat![[x as f32], [y as f32], [1.0]];
                            let tt = h.clone() * tp;
                            (tt[(0, 0)] / tt[(2, 0)], tt[(1, 0)] / tt[(2, 0)])
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
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
