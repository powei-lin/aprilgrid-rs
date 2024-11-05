# aprilgrid-rs

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
```

## Example
```sh
cargo run --example demo -r
```

## TODO
- [ ] Robustness.
- [ ] Unit tests.

## Reference
- https://github.com/AprilRobotics/apriltag
- https://github.com/ethz-asl/kalibr
- https://github.com/powei-lin/aprilgrid
