//! Hermitian cyclic Jacobi iteration with ascending, orthonormal eigenpairs.
//! An unbatched real input rotates only the two affected rows and columns of
//! each pair, in the same cyclic order, instead of masking the whole matrix.

use super::*;

type Pair = (Value, Value);

pub(super) struct HermitianEigen {
    pub(super) values: Value,
    pub(super) vectors: Option<Pair>,
}

impl Emitter<'_> {
    fn hermitian_adjoint(&mut self, a: &Pair) -> Pair {
        let re = self.svd_transpose(&a.0);
        let im = self.svd_transpose(&a.1);
        (re, self.neg(&im))
    }

    fn hermitian_canonical_phase(&mut self, vectors: &Pair) -> Pair {
        let rt = self.svd_transpose(&vectors.0);
        let it = self.svd_transpose(&vectors.1);
        let rr = self.mul(&rt, &rt);
        let ii = self.mul(&it, &it);
        let magnitude = self.add(&rr, &ii);
        let largest = self.reduce_max_last_axis(&magnitude);
        let largest = self.svd_expand(&largest, &rt, false);
        let selected = self.compare("EQ", &magnitude, &largest);
        let indices = self.axis_indices(&rt, self.batch_rank(&rt) + 1);
        let limit = self.int_value_const(i32::MAX as i64);
        let candidates = self.select(&selected, &indices, &limit);
        let pivot = self.reduce_axis_lit(
            "stablehlo.minimum",
            &i32::MAX.to_string(),
            &candidates,
            self.batch_rank(&rt) + 1,
        );
        let pivot = self.svd_expand(&pivot, &rt, false);
        let selected = self.compare("EQ", &indices, &pivot);
        let zero = self.constant_like(0.0, &rt);
        let pr = self.select(&selected, &rt, &zero);
        let pi = self.select(&selected, &it, &zero);
        let pr = self.reduce_sum_last_axis(&pr);
        let pi = self.reduce_sum_last_axis(&pi);
        let rr = self.mul(&pr, &pr);
        let ii = self.mul(&pi, &pi);
        let norm = self.add(&rr, &ii);
        let norm = self.sqrt(&norm);
        let pr = self.div(&pr, &norm);
        let pi = self.div(&pi, &norm);
        let pr = self.svd_expand(&pr, &vectors.0, true);
        let pi = self.svd_expand(&pi, &vectors.0, true);
        let rr = self.mul(&vectors.0, &pr);
        let ii = self.mul(&vectors.1, &pi);
        let ri = self.mul(&vectors.1, &pr);
        let ir = self.mul(&vectors.0, &pi);
        (self.add(&rr, &ii), self.sub(&ri, &ir))
    }

    fn hermitian_entry(&mut self, a: &Value, p: &Value, q: &Value) -> Value {
        let column = self.svd_column(a, q);
        let rows = self.axis_indices(&column, self.batch_rank(&column));
        let selected = self.compare("EQ", &rows, p);
        let zero = self.constant_like(0.0, &column);
        let value = self.select(&selected, &column, &zero);
        self.reduce_sum_last_axis(&value)
    }

    /// A tensor known to hold only zeros, such as the imaginary part of a real input.
    fn known_zero(&self, value: &Value) -> bool {
        if let Some(data) = self.constants.get(&value.ssa) {
            return data
                .iter()
                .all(|x| matches!(x, Scalar::Real(z) if *z == 0.0) || matches!(x, Scalar::Int(0)));
        }
        match self.pointwise.get(&value.ssa).map(|record| &record.op) {
            Some(Pointwise::Reshape(input) | Pointwise::Broadcast(input, _)) => {
                self.known_zero(input)
            }
            _ => false,
        }
    }

    /// A static `rows` x `cols` window of an unbatched matrix at runtime offsets.
    fn matrix_window(
        &mut self,
        matrix: &Value,
        row: &Value,
        col: &Value,
        rows: u64,
        cols: u64,
    ) -> Value {
        let ty = matrix.ty.render(self.dtype, matrix.elem);
        let index_ty = row.ty.render(self.dtype, row.elem);
        let window = MlirTy::Ranked(vec![Some(rows), Some(cols)]);
        let window_ty = window.render(self.dtype, matrix.elem);
        let ssa = self.pure(format!(
            "stablehlo.dynamic_slice {}, {}, {}, sizes = [{rows}, {cols}] : ({ty}, {index_ty}, {index_ty}) -> {window_ty}",
            matrix.ssa, row.ssa, col.ssa,
        ));
        Value {
            ssa,
            ty: window,
            elem: matrix.elem,
        }
    }

    /// Write `update` into an unbatched matrix at runtime offsets.
    fn matrix_write(&mut self, matrix: &Value, update: &Value, row: &Value, col: &Value) -> Value {
        let ty = matrix.ty.render(self.dtype, matrix.elem);
        let update_ty = update.ty.render(self.dtype, update.elem);
        let index_ty = row.ty.render(self.dtype, row.elem);
        let ssa = self.pure(format!(
            "stablehlo.dynamic_update_slice {}, {}, {}, {} : ({ty}, {update_ty}, {index_ty}, {index_ty}) -> {ty}",
            matrix.ssa, update.ssa, row.ssa, col.ssa,
        ));
        let result = Value {
            ssa,
            ..matrix.clone()
        };
        self.axes_view(&result, self.axes_of(matrix))
    }

    /// Entry `index` of a static rank-1 integer table.
    fn table_entry(&mut self, table: &Value, index: &Value) -> Value {
        let ty = table.ty.render(self.dtype, table.elem);
        let index_ty = index.ty.render(self.dtype, index.elem);
        let one = MlirTy::Ranked(vec![Some(1)]);
        let one_ty = one.render(self.dtype, table.elem);
        let ssa = self.pure(format!(
            "stablehlo.dynamic_slice {}, {}, sizes = [1] : ({ty}, {index_ty}) -> {one_ty}",
            table.ssa, index.ssa,
        ));
        let entry = Value {
            ssa,
            ty: one,
            elem: table.elem,
        };
        self.reshape(&entry, MlirTy::Scalar)
    }

    /// Cyclic Jacobi for an unbatched real symmetric matrix: the same pairs,
    /// order and rotations as the general loop, visiting only `p < q` and
    /// updating only columns and rows `p` and `q`, so each pair costs O(n)
    /// instead of O(n^2). Returns the real and imaginary parts of the rotated
    /// matrix and, when requested, of the vectors.
    fn symmetric_jacobi(
        &mut self,
        n: u64,
        a: Value,
        vectors: Option<Value>,
        epsilon: f64,
    ) -> Vec<Value> {
        let pairs: Vec<(i64, i64)> = (0..n as i64)
            .flat_map(|p| (p + 1..n as i64).map(move |q| (p, q)))
            .collect();
        let has_vectors = vectors.is_some();
        let mut initial = vec![a];
        initial.extend(vectors);
        let result = if pairs.is_empty() {
            initial
        } else {
            let table = |e: &mut Self, values: Vec<i64>| {
                let ty = MlirTy::Ranked(vec![Some(values.len() as u64)]);
                let text = values
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let rendered = ty.render(e.dtype, ElemKind::Int);
                let ssa = e.pure(format!("stablehlo.constant dense<[{text}]> : {rendered}"));
                Value {
                    ssa,
                    ty,
                    elem: ElemKind::Int,
                }
            };
            let ps = table(self, pairs.iter().map(|pair| pair.0).collect());
            let qs = table(self, pairs.iter().map(|pair| pair.1).collect());
            let count = pairs.len() as u64;
            self.matrix_loop(32, 0, &initial, |e, _, state| {
                e.matrix_loop(count, 0, state, |e, k, state| {
                    let a = state[0].clone();
                    let p = e.table_entry(&ps, k);
                    let q = e.table_entry(&qs, k);
                    let origin = e.int_value_const(0);
                    let entry = |e: &mut Self, row: &Value, col: &Value| {
                        let window = e.matrix_window(&a, row, col, 1, 1);
                        e.reshape(&window, MlirTy::Scalar)
                    };
                    let alpha = entry(e, &p, &p);
                    let beta = entry(e, &q, &q);
                    let gamma = entry(e, &p, &q);
                    let zero = e.scalar(0.0);
                    let one = e.scalar(1.0);
                    let gamma_scale = e.abs(&gamma);
                    let nonzero = e.compare("GT", &gamma_scale, &zero);
                    let safe = e.select(&nonzero, &gamma_scale, &one);
                    let gr = e.div(&gamma, &safe);
                    let norm2 = e.mul(&gr, &gr);
                    let absolute = e.svd_safe_sqrt(&norm2);
                    let absolute = e.mul(&absolute, &gamma_scale);
                    let eps = e.scalar(epsilon);
                    let active = e.compare("GT", &absolute, &eps);
                    let safe_gamma = e.select(&active, &absolute, &one);
                    let difference = e.sub(&beta, &alpha);
                    let two = e.scalar(2.0);
                    let twice_gamma = e.mul(&two, &safe_gamma);
                    let magnitude = e.abs(&difference);
                    let rotation_scale = e.max(&magnitude, &twice_gamma);
                    let d = e.div(&magnitude, &rotation_scale);
                    let g = e.div(&twice_gamma, &rotation_scale);
                    let d2 = e.mul(&d, &d);
                    let g2 = e.mul(&g, &g);
                    let radius = e.add(&d2, &g2);
                    let radius = e.sqrt(&radius);
                    let denominator = e.add(&d, &radius);
                    let tangent = e.div(&g, &denominator);
                    let negative = e.compare("LT", &difference, &zero);
                    let minus_tangent = e.neg(&tangent);
                    let tangent = e.select(&negative, &minus_tangent, &tangent);
                    let tangent = e.select(&active, &tangent, &zero);
                    let t2 = e.mul(&tangent, &tangent);
                    let norm = e.add(&one, &t2);
                    let norm = e.sqrt(&norm);
                    let cosine = e.div(&one, &norm);
                    let sine = e.mul(&cosine, &tangent);
                    let phase = e.div(&gamma, &safe_gamma);
                    let w = e.mul(&sine, &phase);
                    // p' = c p - w q; q' = w p + c q, as the general rotation with real phase.
                    let rotate = |e: &mut Self, x: &Value, y: &Value| {
                        let cx = e.mul(&cosine, x);
                        let wy = e.mul(&w, y);
                        let wx = e.mul(&w, x);
                        let cy = e.mul(&cosine, y);
                        (e.sub(&cx, &wy), e.add(&wx, &cy))
                    };
                    let columns = |e: &mut Self, m: &Value| {
                        let cp = e.matrix_window(m, &origin, &p, n, 1);
                        let cq = e.matrix_window(m, &origin, &q, n, 1);
                        let (np, nq) = rotate(e, &cp, &cq);
                        let m = e.matrix_write(m, &np, &origin, &p);
                        e.matrix_write(&m, &nq, &origin, &q)
                    };
                    // A' = G^T (A G): columns first, then rows of the result.
                    let a = columns(e, &a);
                    let rp = e.matrix_window(&a, &p, &origin, 1, n);
                    let rq = e.matrix_window(&a, &q, &origin, 1, n);
                    let (np, nq) = rotate(e, &rp, &rq);
                    let a = e.matrix_write(&a, &np, &p, &origin);
                    let a = e.matrix_write(&a, &nq, &q, &origin);
                    let mut next = vec![a];
                    if has_vectors {
                        next.push(columns(e, &state[1]));
                    }
                    next
                })
            })
        };
        let mut result = result.into_iter();
        let a = result.next().unwrap();
        let zero = self.constant_like(0.0, &a);
        let mut out = vec![a, zero];
        if has_vectors {
            let v = result.next().unwrap();
            let zero = self.constant_like(0.0, &v);
            out.extend([v, zero]);
        }
        out
    }

    pub(super) fn matrix_hermitian(
        &mut self,
        id: NodeId,
        input: Pair,
        vectors: bool,
    ) -> Result<HermitianEigen, EmitError> {
        let MlirTy::Ranked(dims) = self.cell_ty(&input.0) else {
            return Err(EmitError::at(
                id,
                "Hermitian eigensolve requires a static square matrix",
            ));
        };
        let [Some(n), Some(m)] = dims.as_slice() else {
            return Err(EmitError::at(
                id,
                "Hermitian eigensolve requires static dimensions",
            ));
        };
        if n != m || *n > i32::MAX as u64 {
            return Err(EmitError::at(
                id,
                "Hermitian eigensolve requires a square matrix",
            ));
        }
        let n = *n;
        let input = (
            self.convert(&input.0, ElemKind::Real),
            self.convert(&input.1, ElemKind::Real),
        );
        let axes = self.axes_of(&input.0);
        let batch = self.batch_rank(&input.0);
        let rows = self.axis_indices(&input.0, batch);
        let cols = self.axis_indices(&input.0, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let zero_matrix = self.constant_like(0.0, &input.0);
        if n == 0 {
            let values = self.diagonal_axes(&input.0, batch, batch + 1);
            return Ok(HermitianEigen {
                values,
                vectors: vectors.then_some((identity, zero_matrix)),
            });
        }
        let epsilon = match self.dtype {
            Dtype::F32 => f32::EPSILON as f64,
            Dtype::F64 => f64::EPSILON,
        };
        let ar = self.abs(&input.0);
        let ai = self.abs(&input.1);
        let magnitude = self.max(&ar, &ai);
        let scale = self.reduce_max(&magnitude);
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let positive = self.compare("GT", &scale, &zero);
        let safe_scale = self.select(&positive, &scale, &one);
        let scaled = (
            self.div(&input.0, &safe_scale),
            self.div(&input.1, &safe_scale),
        );
        let adjoint = self.hermitian_adjoint(&scaled);
        let dr = self.sub(&scaled.0, &adjoint.0);
        let di = self.sub(&scaled.1, &adjoint.1);
        let dr = self.abs(&dr);
        let di = self.abs(&di);
        let error = self.max(&dr, &di);
        let error = self.reduce_max(&error);
        let tolerance = self.scalar(64.0 * epsilon * n as f64);
        let hermitian = self.compare("LE", &error, &tolerance);
        self.record_runtime_error(&hermitian, 12);
        let half = self.scalar(0.5);
        let re = self.add(&scaled.0, &adjoint.0);
        let im = self.add(&scaled.1, &adjoint.1);
        let re = self.mul(&half, &re);
        let im = self.mul(&half, &im);
        let real = batch == 0 && self.known_zero(&input.1);
        let mut initial = vec![re, im];
        if vectors {
            initial.extend([identity.clone(), zero_matrix]);
        }
        let result = if real {
            let vectors = vectors.then(|| initial[2].clone());
            self.symmetric_jacobi(n, initial[0].clone(), vectors, epsilon)
        } else {
            self.matrix_loop(32, 0, &initial, |e, _, state| {
                e.matrix_loop(n * n, 0, state, |e, index, state| {
                    let state: Vec<_> =
                        state.iter().map(|v| e.axes_view(v, axes.clone())).collect();
                    let size = e.int_value_const(n as i64);
                    let p = e.trunc_div(index, &size);
                    let q = e.rem(index, &size);
                    let a = (state[0].clone(), state[1].clone());
                    let alpha = e.hermitian_entry(&a.0, &p, &p);
                    let beta = e.hermitian_entry(&a.0, &q, &q);
                    let gamma = (
                        e.hermitian_entry(&a.0, &p, &q),
                        e.hermitian_entry(&a.1, &p, &q),
                    );
                    let gr = e.abs(&gamma.0);
                    let gi = e.abs(&gamma.1);
                    let gamma_scale = e.max(&gr, &gi);
                    let nonzero = e.compare("GT", &gamma_scale, &zero);
                    let safe = e.select(&nonzero, &gamma_scale, &one);
                    let gr = e.div(&gamma.0, &safe);
                    let gi = e.div(&gamma.1, &safe);
                    let gr2 = e.mul(&gr, &gr);
                    let gi2 = e.mul(&gi, &gi);
                    let norm2 = e.add(&gr2, &gi2);
                    let absolute = e.svd_safe_sqrt(&norm2);
                    let absolute = e.mul(&absolute, &gamma_scale);
                    let eps = e.scalar(epsilon);
                    let correlated = e.compare("GT", &absolute, &eps);
                    let ordered = e.compare("LT", &p, &q);
                    let active = e.and(&ordered, &correlated);
                    let safe_gamma = e.select(&active, &absolute, &one);
                    let difference = e.sub(&beta, &alpha);
                    let two = e.scalar(2.0);
                    let twice_gamma = e.mul(&two, &safe_gamma);
                    let magnitude = e.abs(&difference);
                    let rotation_scale = e.max(&magnitude, &twice_gamma);
                    let d = e.div(&magnitude, &rotation_scale);
                    let g = e.div(&twice_gamma, &rotation_scale);
                    let d2 = e.mul(&d, &d);
                    let g2 = e.mul(&g, &g);
                    let radius = e.add(&d2, &g2);
                    let radius = e.sqrt(&radius);
                    let denominator = e.add(&d, &radius);
                    let tangent = e.div(&g, &denominator);
                    let negative = e.compare("LT", &difference, &zero);
                    let minus_tangent = e.neg(&tangent);
                    let tangent = e.select(&negative, &minus_tangent, &tangent);
                    let tangent = e.select(&active, &tangent, &zero);
                    let t2 = e.mul(&tangent, &tangent);
                    let norm = e.add(&one, &t2);
                    let norm = e.sqrt(&norm);
                    let cosine = e.div(&one, &norm);
                    let sine = e.mul(&cosine, &tangent);
                    let phase = (e.div(&gamma.0, &safe_gamma), e.div(&gamma.1, &safe_gamma));
                    let right = e.svd_rotate(&a, &p, &q, &cosine, &sine, &phase);
                    let adjoint = e.hermitian_adjoint(&right);
                    let both = e.svd_rotate(&adjoint, &p, &q, &cosine, &sine, &phase);
                    let both = e.hermitian_adjoint(&both);
                    let mut next = vec![both.0, both.1];
                    if vectors {
                        let v = e.svd_rotate(
                            &(state[2].clone(), state[3].clone()),
                            &p,
                            &q,
                            &cosine,
                            &sine,
                            &phase,
                        );
                        next.extend([v.0, v.1]);
                    }
                    next
                })
            })
        };
        let state: Vec<_> = result
            .iter()
            .map(|v| self.axes_view(v, axes.clone()))
            .collect();
        let values = self.diagonal_axes(&state[0], batch, batch + 1);
        let zero_matrix = self.constant_like(0.0, &state[0]);
        let off_r = self.select(&diagonal, &zero_matrix, &state[0]);
        let off_i = self.select(&diagonal, &zero_matrix, &state[1]);
        let off_r = self.abs(&off_r);
        let off_i = self.abs(&off_i);
        let off = self.max(&off_r, &off_i);
        let error = self.reduce_max(&off);
        let converged = self.compare("LE", &error, &tolerance);
        self.record_runtime_error(&converged, 10);
        let valid = self.and(&hermitian, &converged);
        // Stable sorting by pairwise rank retains one column per equal value.
        let column_values = self.svd_expand(&values, &identity, false);
        let row_values = self.svd_expand(&values, &identity, true);
        let less = self.compare("LT", &column_values, &row_values);
        let equal = self.compare("EQ", &column_values, &row_values);
        let before = self.compare("LT", &rows, &cols);
        let tied = self.and(&equal, &before);
        let earlier = self.or(&less, &tied);
        let earlier = self.convert(&earlier, ElemKind::Int);
        let ranks = self.reduce_axis("stablehlo.add", "0", &earlier, batch);
        let ranks = self.svd_expand(&ranks, &identity, false);
        let sorted = self.compare("EQ", &ranks, &cols);
        let permutation = self.convert(&sorted, ElemKind::Real);
        let permutation_t = self.svd_transpose(&permutation);
        let values = self.mul(&values, &scale);
        let values = self.matvec(&permutation_t, &values);
        let nan = self.scalar(f64::NAN);
        let values = self.select(&valid, &values, &nan);
        let vectors = if vectors {
            let re = self.matmat(&state[2], &permutation);
            let im = self.matmat(&state[3], &permutation);
            let (re, im) = self.hermitian_canonical_phase(&(re, im));
            Some((
                self.select(&valid, &re, &nan),
                self.select(&valid, &im, &nan),
            ))
        } else {
            None
        };
        Ok(HermitianEigen { values, vectors })
    }
}
