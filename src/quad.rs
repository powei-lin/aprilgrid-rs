use faer::solvers::SpSolver;
use geo::{Area, LineString, MinimumRotatedRect, Polygon};
use image::DynamicImage;
use imageproc::{
    contours::{find_contours, BorderType},
    morphology::dilate,
};

fn distance2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
}

fn get_option_edge_indexes(
    quad_indexes: &[usize],
    total_idx: usize,
    min_edge_points: usize,
) -> Option<Vec<Vec<usize>>> {
    let mut edge_indexes = vec![Vec::<usize>::new(); 4];
    if quad_indexes.len() != 4 {
        return None;
    }
    let mut quad_indexes: Vec<_> = quad_indexes.into();
    quad_indexes.sort();
    let mut slot = 0;
    for i in 0..total_idx {
        if i == quad_indexes[slot % 4] {
            slot += 1;
        } else {
            edge_indexes[slot % 4].push(i);
        }
    }
    if edge_indexes.iter().any(|v| v.len() < min_edge_points) {
        None
    } else {
        Some(edge_indexes)
    }
}

/// Find ax + by + c = 0
/// return (a, b, c)
fn find_line_svd(points: &[(f32, f32)]) -> (f32, f32, f32) {
    let mut mat_a = faer::Mat::<f32>::zeros(points.len(), 2);

    for (i, (px, py)) in points.iter().enumerate() {
        unsafe {
            mat_a.write_unchecked(i, 0, *px);
            mat_a.write_unchecked(i, 1, *py);
        }
    }
    let mean_x = mat_a.col(0).sum() / points.len() as f32;
    let mean_y = mat_a.col(1).sum() / points.len() as f32;
    mat_a.col_mut(0).iter_mut().for_each(|i| *i -= mean_x);
    mat_a.col_mut(1).iter_mut().for_each(|i| *i -= mean_y);
    let svd = mat_a.svd();
    let h = svd.v().col(1);

    let a = h[0];
    let b = h[1];
    let c = -(a * mean_x + b * mean_y);

    (a, b, c)
}

type VecPointsf32 = Vec<(f32, f32)>;
fn min_quad_from_p2ds(p2ds: &Vec<(f32, f32)>) -> ((Vec<usize>, VecPointsf32), Polygon<f32>) {
    let ls = LineString::<f32>::from(p2ds.to_owned());
    let poly = Polygon::new(ls, vec![]);
    let mbr = MinimumRotatedRect::minimum_rotated_rect(&poly).unwrap();
    (
        mbr.exterior().0[..4]
            .iter()
            .map(|m| {
                let (mx, my) = m.x_y();
                let (idx, cor) = p2ds.iter().enumerate().fold(
                    (usize::MAX, (0.0, 0.0)),
                    |(best_idx, (best_x, best_y)), (cur_idx, (cur_x, cur_y))| {
                        if cur_idx == 0
                            || distance2(mx, my, *cur_x, *cur_y) < distance2(mx, my, best_x, best_y)
                        {
                            (cur_idx, (*cur_x, *cur_y))
                        } else {
                            (best_idx, (best_x, best_y))
                        }
                    },
                );
                (idx, cor)
            })
            .unzip(),
        mbr,
    )
}

pub fn find_quad(img: &DynamicImage, min_area: f32) -> Vec<Vec<(f32, f32)>> {
    let img0_grey = img.adjust_contrast(2000.0);
    let max_pool = dilate(
        &img0_grey.to_luma8(),
        imageproc::distance_transform::Norm::LInf,
        2,
    );
    let contours = find_contours::<u32>(&max_pool);

    let quads: Vec<Vec<(f32, f32)>> = contours
        .iter()
        .filter_map(|c| {
            if c.points.len() < 4 || c.border_type != BorderType::Hole {
                return None;
            }
            let p2ds: Vec<(f32, f32)> = c.points.iter().map(|p| (p.x as f32, p.y as f32)).collect();
            let ((min_quad_indexes, min_quad), mbr) = min_quad_from_p2ds(&p2ds);

            let final_poly = Polygon::new(LineString::from(min_quad), vec![]);
            let pa = final_poly.unsigned_area();
            if pa < min_area || pa / mbr.unsigned_area() < 0.75 {
                return None;
            }

            let edge_indexes_option = get_option_edge_indexes(&min_quad_indexes, p2ds.len(), 3);
            if let Some(edge_indexes) = edge_indexes_option {
                let line_abc: Vec<(f32, f32, f32)> = edge_indexes
                    .iter()
                    .map(|edge_inds| {
                        let edge_pts: Vec<(f32, f32)> =
                            edge_inds.iter().map(|idx| p2ds[*idx]).collect();
                        find_line_svd(&edge_pts)
                    })
                    .collect();
                let intersect_points: Vec<(f32, f32)> = (0..4)
                    .map(|i| {
                        let (a0, b0, c0) = line_abc[i];
                        let (a1, b1, c1) = line_abc[(i + 1) % 4];
                        let a = faer::mat![[a0, b0], [a1, b1]];
                        let b = faer::mat![[-c0], [-c1]];
                        let plu = a.partial_piv_lu();
                        let x1 = plu.solve(&b);

                        unsafe { (*x1.get_unchecked(0, 0), *x1.get_unchecked(1, 0)) }
                    })
                    .collect();
                Some(intersect_points)
            } else {
                None
            }
        })
        .collect();
    quads
}