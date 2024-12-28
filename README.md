# aprilgrid-rs
[![crate](https://img.shields.io/crates/v/aprilgrid.svg)](https://crates.io/crates/aprilgrid)

### Pure Rust version of aprilgrid


<img src="docs/example.png" width="800" alt="example detection">

## Install from cargo
```
cargo add aprilgrid
```

## Usage
See examples/demo.rs

```rust
// load image
let img = ImageReader::open(path.unwrap())?.decode()?;

// create detector
let detector = aprilgrid::detector::TagDetector::new(&aprilgrid::TagFamily::T36H11, None);

// detect tags
let tags = detector.detect(&img);

// support kornia image
// cargo add aprilgrid -F kornia
let image: kornia::image::Image<u8, 3> =
    kornia::io::functional::read_image_any("...").unwrap();
let tags = detector.detect_kornia(&image);
```

## Example
```sh
cargo run --example demo -r
```

## Run tests
```sh
cargo test -r
```

## TODO
- [ ] Robustness.
- [x] Unit tests.

## Generate chart pdf
```sh
pip install opencv-python pillow cairosvg svgwrite
python3 scripts/generate_aprilgrid.py -h
```

## Reference
- https://github.com/AprilRobotics/apriltag
- https://github.com/ethz-asl/kalibr
- https://github.com/powei-lin/aprilgrid
