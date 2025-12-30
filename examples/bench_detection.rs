use aprilgrid::{TagFamily, detector::TagDetector};
use diol::prelude::*;
use image::ImageReader;

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register(
        "detection",
        bench_detection,
        [
            "tests/data/iphone.png",
            "tests/data/EuRoC.png",
            "tests/data/TUM_VI.png",
            "tests/data/right.png",
            "tests/data/r45.png",
            "tests/data/top.png",
            "tests/data/two_boards.png",
        ],
    );
    bench.run()?;
    Ok(())
}

fn bench_detection(bencher: Bencher, img_path: &str) {
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
