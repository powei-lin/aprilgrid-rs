#[cfg(test)]
mod tests {
    use aprilgrid::{TagFamily, detector};
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
                println!("{}", tags.len());
                assert!(tags.len() == expected);
            }
        )*
        }
    }

    detector_tests! {
        detect_iphone: (TagFamily::T36H11, "tests/data/iphone.png", 66),
        detect_euroc: (TagFamily::T36H11, "tests/data/EuRoC.png", 36),
        detect_tum_vi: (TagFamily::T36H11, "tests/data/TUM_VI.png", 36),
        detect_tum_vi_right: (TagFamily::T36H11, "tests/data/right.png", 36),
        detect_tum_vi_r45: (TagFamily::T36H11, "tests/data/r45.png", 36),
        detect_tum_vi_top: (TagFamily::T36H11, "tests/data/top.png", 36),
        detect_two_boards: (TagFamily::T36H11, "tests/data/two_boards.png", 72),
    }

    #[test]
    #[cfg(feature = "kornia")]
    fn test_kornia() {
        let image: kornia::image::Image<u8, 3> =
            kornia::io::functional::read_image_any("tests/data/iphone.png").unwrap();
        let detector = detector::TagDetector::new(&TagFamily::T36H11, None);
        let tags = detector.detect_kornia(&image);
        assert!(tags.len() == 66);
    }
}
