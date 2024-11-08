#[cfg(test)]
mod tests {
    use aprilgrid::{detector, TagFamily};
    use image::ImageReader;

    macro_rules! detector_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (tag_family, img_path, expected) = $value;
                let detector = detector::TagDetector::new(&tag_family, None);
                let img = ImageReader::open(img_path)
                    .unwrap()
                    .decode()
                    .unwrap();
                let tags = detector.detect(&img);
                assert!(tags.len() == expected);
            }
        )*
        }
    }

    detector_tests! {
        detect_iphone: (TagFamily::T36H11, "tests/data/iphone.png", 66),
        detect_euroc: (TagFamily::T36H11, "tests/data/EuRoC.png", 36),
        detect_tum_vi: (TagFamily::T36H11, "tests/data/TUM_VI.png", 36),
    }
}
