use core::f32;
use glob::glob;
use image::{
    imageops::FilterType::{Nearest, Triangle},
    DynamicImage, GenericImage, GenericImageView, GrayImage, ImageReader,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::RecordingStream;
use std::io::Cursor;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    // let mut time_sec = 0.0;
    // let fps = 60.0;
    // let one_frame_time = 1.0 / fps;
    // let detector_params = None;
    let img_path = "tests/data/1520525772624130511.png";
    let img = ImageReader::open(img_path)?.decode()?;
    // let small = image::imageops::resize(&img.to_luma32f(), 100, 100, Triangle);
    let blur: image::ImageBuffer<image::Luma<f32>, Vec<f32>> =
        imageproc::filter::gaussian_blur_f32(&img.to_luma32f(), 1.5);
    let h = hessian_response(&blur);
    // let mm = h.to_vec().iter().fold(f32::MIN, |acc, e|{acc.max(*e)});
    let mn = h.to_vec().iter().fold(f32::MAX, |acc, e| acc.min(*e));
    let init_saddle_points: Vec<_> = h
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if *p < mn * 0.05 {
                Some((i as u32 % h.width(), i as u32 / h.width()))
            } else {
                None
            }
        })
        .collect();
    let mv = h
        .to_vec()
        .iter()
        .map(|p| if *p < mn * 0.05 { 255 } else { 0 })
        .collect();
    let f32pts: Vec<_> = init_saddle_points
        .iter()
        .map(|(x, y)| (*x as f32, *y as f32))
        .collect();
    let m = GrayImage::from_vec(h.width(), h.height(), mv).unwrap();
    let clusters = init_saddle_clusters(&h, mn * 0.05);
    let cs: Vec<(f32, f32)> = clusters
        .iter()
        .map(|c| {
            let (sx, sy) = c.iter().fold((0.0, 0.0), |(ax, ay), (ex, ey)| {
                (ax + *ex as f32, ay + *ey as f32)
            });
            (sx / c.len() as f32, sy / c.len() as f32)
        })
        .collect();
    // let m = imageproc::morphology::dilate(&m, imageproc::distance_transform::Norm::L1, 1);

    // let b = GrayImage::

    // let img_grey = imageproc::contrast::adaptive_threshold(&blur, 2);
    // let img_d = imageproc::morphology::dilate(&img_grey, imageproc::distance_transform::Norm::L1, 1);
    // let edge = imageproc::edges::canny(&blur, 10.0, 50.0);

    // recording.set_time_nanos("stable_time", time_ns);
    // recording.set_time_seconds("stable_time", time_sec);
    // time_sec += one_frame_time;

    log_image_as_compressed(&recording, "/cam0", &img);
    log_grey_image_as_compressed(&recording, "/cam0", &m);
    recording
        .log(
            format!("/cam0/image/corners"),
            &rerun::Points2D::new(rerun_shift(&f32pts))
                .with_radii([rerun::Radius::new_ui_points(1.0)]),
        )
        .expect("msg");
    recording
        .log(
            format!("/cam0/image/cluster"),
            &rerun::Points2D::new(rerun_shift(&cs)).with_radii([rerun::Radius::new_ui_points(1.0)]),
        )
        .expect("msg");
    // log_grey_image_as_compressed(&recording, "/cam0", &img_d);
    // log_grey_image_as_compressed(&recording, "/edge", &edge);

    Ok(())
}
