use crate::saddle::{is_valid_quad, Saddle};
use glam;
use kiddo::{KdTree, SquaredEuclidean};
use std::collections::{HashMap, HashSet};

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct BoardIdx {
    x: i32,
    y: i32,
}
impl BoardIdx {
    pub fn new(x: i32, y: i32) -> BoardIdx {
        BoardIdx { x, y }
    }
}

pub struct Board<'a> {
    refined: &'a [Saddle],
    active_idxs: HashSet<usize>,
    found_board_idxs: HashMap<BoardIdx, Option<[usize; 4]>>,
    tree: KdTree<f32, 2>,
    spacing_ratio: f32,
    pub score: u32,
}
impl<'a> Board<'a> {
    pub fn new(
        refined: &'a [Saddle],
        active_idxs: &HashSet<usize>,
        quad_idxs: &[usize; 4],
        spacing_ratio: f32,
        tree: &KdTree<f32, 2>,
    ) -> Board<'a> {
        let mut active_idxs = active_idxs.clone();
        let mut tree = tree.clone();
        for i in &quad_idxs[1..] {
            active_idxs.remove(i);
            tree.remove(&refined[*i].arr(), *i as u64);
        }
        let mut b = Board {
            refined,
            active_idxs,
            found_board_idxs: HashMap::from([(BoardIdx::new(0, 0), Some(*quad_idxs))]),
            tree,
            spacing_ratio,
            score: 1,
        };
        b.try_expand(&BoardIdx::new(0, 0));
        b
    }
    pub fn all_tag_indexes(&self) -> Vec<[usize; 4]> {
        self.found_board_idxs.values().filter_map(|i| *i).collect()
    }
    pub fn try_fix_missing(&mut self) {
        let fix_list: Vec<(BoardIdx, BoardIdx)> = self
            .found_board_idxs
            .iter()
            .filter_map(|(bid, result)| {
                if result.is_none() {
                    let b0 = BoardIdx::new(bid.x + 1, bid.y);
                    let b1 = BoardIdx::new(bid.x - 1, bid.y);
                    let b2 = BoardIdx::new(bid.x, bid.y + 1);
                    let b3 = BoardIdx::new(bid.x, bid.y - 1);
                    if self.found_board_idxs.contains_key(&b0)
                        && self.found_board_idxs.contains_key(&b1)
                    {
                        if self.found_board_idxs.get(&b0).unwrap().is_some()
                            && self.found_board_idxs.get(&b1).unwrap().is_some()
                        {
                            return Some((b0, b1));
                        }
                    } else if self.found_board_idxs.contains_key(&b2)
                        && self.found_board_idxs.contains_key(&b3)
                    {
                        if self.found_board_idxs.get(&b2).unwrap().is_some()
                            && self.found_board_idxs.get(&b3).unwrap().is_some()
                        {
                            return Some((b2, b3));
                        }
                    }
                }
                None
            })
            .collect();
        for (b0, b1) in fix_list {
            let q0 = self.found_board_idxs.get(&b0).unwrap().unwrap();
            let q1 = self.found_board_idxs.get(&b1).unwrap().unwrap();
            let saddle_idxs: Vec<usize> = (0..4)
                .map(|i| {
                    let x = (self.refined[q0[i]].p.0 + self.refined[q1[i]].p.0) / 2.0;
                    let y = (self.refined[q0[i]].p.1 + self.refined[q1[i]].p.1) / 2.0;
                    let n = self.tree.nearest_one::<SquaredEuclidean>(&[x, y]);
                    n.item as usize
                })
                .collect();
            // println!("try fixing {:?} {:?}", b0, b1);
            if is_valid_quad(
                &self.refined[saddle_idxs[0]],
                &self.refined[saddle_idxs[1]],
                &self.refined[saddle_idxs[2]],
                &self.refined[saddle_idxs[3]],
            ) {
                let b = BoardIdx::new((b0.x + b1.x) / 2, (b0.y + b1.y) / 2);
                // println!("suc");
                self.found_board_idxs.insert(
                    b,
                    Some([
                        saddle_idxs[0],
                        saddle_idxs[1],
                        saddle_idxs[2],
                        saddle_idxs[3],
                    ]),
                );
            }
        }
    }

    fn try_expand(&mut self, board_idx: &BoardIdx) {
        let start_board = self.found_board_idxs.get(board_idx).unwrap().to_owned();
        if let Some(quad_idxs) = start_board {
            for i in 0..4 {
                let mut qs: Vec<usize> = quad_idxs.to_vec();
                qs.rotate_left(i);
                let new_board_idx = match i {
                    0 => BoardIdx::new(board_idx.x + 1, board_idx.y),
                    1 => BoardIdx::new(board_idx.x, board_idx.y - 1),
                    2 => BoardIdx::new(board_idx.x - 1, board_idx.y),
                    3 => BoardIdx::new(board_idx.x, board_idx.y + 1),
                    _ => {
                        panic!("should not reach");
                    }
                };

                // TODO need review
                if self.found_board_idxs.contains_key(&new_board_idx)
                    && self.found_board_idxs.get(&new_board_idx).unwrap().is_some()
                {
                    continue;
                }

                if let Some(valid_new_qs) = self.try_expand_one(&qs.try_into().unwrap()) {
                    let mut v = valid_new_qs.to_vec();
                    v.rotate_right(i);
                    for vv in &v {
                        self.active_idxs.remove(vv);
                    }
                    self.score += 1;
                    self.found_board_idxs
                        .insert(new_board_idx, Some(v.try_into().unwrap()));
                    self.try_expand(&new_board_idx);
                } else {
                    self.found_board_idxs.insert(new_board_idx, None);
                }
            }
        }
    }
    fn try_expand_one(&self, quad_idxs: &[usize; 4]) -> Option<[usize; 4]> {
        let s0 = self.refined[quad_idxs[0]];
        let s1 = self.refined[quad_idxs[1]];
        let s2 = self.refined[quad_idxs[2]];
        let s3 = self.refined[quad_idxs[3]];
        let (new_s0s, new_s1s) = self.find_closest_potential_saddle_idxs(&s0, &s1);
        let (new_s3s, new_s2s) = self.find_closest_potential_saddle_idxs(&s3, &s2);
        for s0_i in &new_s0s {
            for s1_i in &new_s1s {
                for s2_i in &new_s2s {
                    for s3_i in &new_s3s {
                        let new_s0 = self.refined[*s0_i];
                        let new_s1 = self.refined[*s1_i];
                        let new_s2 = self.refined[*s2_i];
                        let new_s3 = self.refined[*s3_i];
                        if is_valid_quad(&new_s0, &new_s1, &new_s2, &new_s3) {
                            return Some([*s0_i, *s1_i, *s2_i, *s3_i]);
                        }
                    }
                }
            }
        }
        None
    }
    fn find_closest_potential_saddle_idxs(
        &self,
        s0: &Saddle,
        s1: &Saddle,
    ) -> (Vec<usize>, Vec<usize>) {
        let ratio0 = 1.0 + self.spacing_ratio;
        let ratio1 = 0.5;
        let angle_thres = 5.0;
        let v0 = glam::Vec2::new(s0.p.0, s0.p.1);
        let v1 = glam::Vec2::new(s1.p.0, s1.p.1);
        let v10 = v1 - v0;
        let v10_norm = v10.distance_squared(glam::Vec2::ZERO);
        let new_v0 = v0 + v10 * ratio0;
        let new_v1 = v1 + v10 * ratio0;
        let nv0s = self.tree.nearest_n_within::<SquaredEuclidean>(
            &[new_v0.x, new_v0.y],
            ratio1 * v10_norm,
            3,
            true,
        );
        let new_s0: Vec<usize> = nv0s
            .iter()
            .filter_map(|n| {
                let idx = n.item as usize;
                if self.active_idxs.contains(&idx) {
                    let theta_diff =
                        crate::math_util::theta_distance_degree(s0.theta, self.refined[idx].theta);
                    if theta_diff < angle_thres {
                        return Some(idx);
                    }
                }
                None
            })
            .collect();
        let nv1s = self.tree.nearest_n_within::<SquaredEuclidean>(
            &[new_v1.x, new_v1.y],
            ratio1 * v10_norm,
            3,
            true,
        );
        let new_s1: Vec<usize> = nv1s
            .iter()
            .filter_map(|n| {
                let idx = n.item as usize;
                if self.active_idxs.contains(&idx) {
                    let theta_diff =
                        crate::math_util::theta_distance_degree(s1.theta, self.refined[idx].theta);
                    if theta_diff < angle_thres {
                        return Some(idx);
                    }
                }
                None
            })
            .collect();
        (new_s0, new_s1)
    }
}
