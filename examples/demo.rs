use glob::glob;
use image::imageops::blur;
use image::{DynamicImage, ImageReader};
use imageproc::contours::{find_contours, BorderType};
use imageproc::filter::median_filter;
use imageproc::morphology::{dilate, erode};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rerun::external::glam::vec2;
use rerun::RecordingStream;
use std::io::Cursor;
// primitives
use geo::{
    line_string, polygon, Area, ClosestPoint, LineString, MinimumRotatedRect, Point, Polygon,
    Simplify, SimplifyVw, SimplifyVwPreserve,
};

// algorithms
use geo::ConvexHull;

fn log_image_as_jpeg(recording: &RecordingStream, topic: &str, img: &DynamicImage) {
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
    let color = (
        ((color_num >> 16) % 256) as u8,
        ((color_num >> 8) % 256) as u8,
        (color_num % 256) as u8,
        255,
    );
    color
}

fn polygon_to_vec_pts(poly: &Polygon<f32>, scale: f32) -> Vec<(f32, f32)> {
    poly.exterior()
        .0
        .iter()
        .map(|c| (c.x * scale, c.y * scale))
        .collect()
}

fn coord_distance2(a: &geo::Coord<f32>, b: &geo::Coord<f32>) -> f32 {
    let x = (a.x - b.x);
    let y = (a.y - b.y);
    x * x + y * y
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let recording = rerun::RecordingStreamBuilder::new("aprilgrid").spawn()?;
    let dataset_root = "data";
    let img_paths = glob(format!("{}/*.png", dataset_root).as_str()).expect("failed");

    for path in img_paths {
        let img0 = ImageReader::open(&path.unwrap())?.decode()?;
        let img0_luma8 = img0.to_luma8();
        // let (min_pool, max_pool) = aprilgrid_rs::threshold(&img0_luma8);
        let img0_grey = img0.adjust_contrast(2000.0);
        let max_pool = dilate(
            &img0_grey.to_luma8(),
            imageproc::distance_transform::Norm::LInf,
            2,
        );
        // let max_pool = dilate(&max_pool, imageproc::distance_transform::Norm::L2, 2);
        let img0_grey = DynamicImage::ImageLuma8(max_pool.clone());
        let contours = find_contours::<u32>(&max_pool);

        // let img0_grey = DynamicImage::ImageLuma8(img0_luma8);
        log_image_as_jpeg(&recording, "/cam0", &img0);
        let img0_max = DynamicImage::ImageLuma8(max_pool);
        log_image_as_jpeg(&recording, "/cam0_max", &img0_max);
        for (i, c) in contours.iter().enumerate() {
            if c.points.len() >= 4 && c.border_type == BorderType::Hole {
                let p2ds: Vec<(f32, f32)> =
                    c.points.iter().map(|p| (p.x as f32, p.y as f32)).collect();
                // An L shape
                let ls = LineString::<f32>::from(p2ds.clone());
                let poly = Polygon::new(ls, vec![]);

                // let con = poly.simplify(&3.0);
                let mbr = MinimumRotatedRect::minimum_rotated_rect(&poly).unwrap();
                let min_quad: Vec<(f32, f32)> = mbr
                    .exterior()
                    .0
                    .iter()
                    .map(|m| {
                        let c = poly
                            .exterior()
                            .0
                            .iter()
                            .reduce(|a, b| {
                                if coord_distance2(a, m) < coord_distance2(b, m) {
                                    a
                                } else {
                                    b
                                }
                            })
                            .unwrap();
                        (c.x, c.y)
                    })
                    .collect();

                let final_poly = Polygon::new(LineString::from(min_quad), vec![]);
                let pa = final_poly.unsigned_area();
                if pa < 100.0 || pa / mbr.unsigned_area() < 0.75 {
                    continue;
                }

                let p2ds_small: Vec<(f32, f32)> =
                    final_poly.exterior().0.iter().map(|c| (c.x, c.y)).collect();
                let p2ds2: Vec<_> = p2ds_small.iter().map(|(x, y)| (x * 1.0, y * 1.0)).collect();
                let h = aprilgrid_rs::homography::tag_homography(&p2ds2, 10);
                let homo_points: Vec<(f32, f32)> = (2..8)
                    .map(|x| {
                        (2..8)
                            .map(|y| {
                                let tp = faer::mat![[x as f32], [y as f32], [1.0]];
                                let tt = h.clone() * tp;
                                // println!("tt {:?}", tt);
                                (tt[(0, 0)] / tt[(2, 0)], tt[(1, 0)] / tt[(2, 0)])
                                // (0.0, 0.0)
                            })
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .collect();
                for p in &p2ds_small {
                    println!("({}, {})", p.0, p.1);
                }
                println!("len {}", p2ds_small.len());

                // Calculate the polygon's convex hull
                let mut colors = vec![id_to_color(i); p2ds_small.len()];
                colors[0] = (255, 0, 0, 255);

                let l = rerun::LineStrip2D::from_iter(p2ds_small.clone());
                recording
                    .log(
                        format!("/cam0/p2ds"),
                        &rerun::Points2D::new(p2ds.iter().map(|(x, y)| (x * 1.0, y * 1.0)))
                            .with_colors(colors)
                            .with_radii([rerun::Radius::new_ui_points(3.0)]),
                    )
                    .expect("msg");
                recording
                    .log(
                        format!("/cam0/pts"),
                        &rerun::Points2D::new(homo_points)
                            .with_radii([rerun::Radius::new_ui_points(3.0)]),
                    )
                    .expect("msg");
                recording
                    .log(
                        format!("/cam0/mbr"),
                        &rerun::Points2D::new(polygon_to_vec_pts(&mbr, 1.0))
                            .with_radii([rerun::Radius::new_ui_points(3.0)]),
                    )
                    .expect("msg");
                recording
                    .log(
                        format!("/cam0/mq"),
                        &rerun::Points2D::new(polygon_to_vec_pts(&final_poly, 1.0))
                            .with_radii([rerun::Radius::new_ui_points(3.0)]),
                    )
                    .expect("msg");
                recording
                    .log(
                        format!("/cam0_max/lines"),
                        &rerun::LineStrips2D::new([l])
                            .with_colors(vec![id_to_color(i)])
                            .with_radii([rerun::Radius::new_ui_points(1.0)]),
                    )
                    .expect("msg");
            }
        }
    }

    Ok(())
}
