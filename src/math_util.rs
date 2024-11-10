use faer::solvers::SpSolver;

pub fn find_xy(a0: f32, b0: f32, c0: f32, a1: f32, b1: f32, c1: f32) -> (f32, f32) {
    let a = faer::mat![[a0, b0], [a1, b1]];
    let b = faer::mat![[-c0], [-c1]];
    let plu = a.partial_piv_lu();
    let x1 = plu.solve(&b);

    unsafe { (*x1.get_unchecked(0, 0), *x1.get_unchecked(1, 0)) }
}
