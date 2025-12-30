use aprilgrid::image_util;
use diol::prelude::*;
use image::ImageReader;

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register_many(
        "imageproc_blur",
        list![bench_imageproc_blur, bench_aprilgrid_blur,],
        [
            "tests/data/iphone.png",
            "tests/data/EuRoC.png",
            "tests/data/TUM_VI.png",
        ],
    );
    bench.run()?;
    Ok(())
}

fn bench_imageproc_blur(bencher: Bencher, img_path: &str) {
    let img = ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image");

    let luma_f32 = img.to_luma32f();

    bencher.bench(|| {
        let _blur = imageproc::filter::gaussian_blur_f32(&luma_f32, 1.5);
        black_box(_blur);
    });
}

fn bench_aprilgrid_blur(bencher: Bencher, img_path: &str) {
    let img = ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image");

    let luma_f32 = img.to_luma32f();

    bencher.bench(|| {
        let _blur = image_util::gaussian_blur_f32(&luma_f32, 1.5);
        black_box(_blur);
    });
}
