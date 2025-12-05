use aprilgrid::{TagFamily, detector::TagDetector};
use diol::prelude::*;
use image::ImageReader;

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register("detection", bench_detection, [()]);
    bench.run()?;
    Ok(())
}

fn bench_detection(bencher: Bencher, _: ()) {
    let img_path = "tests/data/iphone.png";
    let img = ImageReader::open(img_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image");

    let detector = TagDetector::new(&TagFamily::T36H11, None);

    bencher.bench(|| {
        let tags = detector.detect(&img);
        black_box(tags);
    });
}
