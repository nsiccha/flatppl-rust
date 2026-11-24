//! Hermitian cyclic Jacobi iteration with ascending, orthonormal eigenpairs.
//! An unbatched real input rotates only the two affected rows and columns of
//! each pair, in the same cyclic order, instead of masking the whole matrix,
//! and is emitted as the versioned private helper `__enzyme_linalg_eigh_v1_*`.

use super::*;

type Pair = (Value, Value);

pub(super) struct HermitianEigen {
    pub(super) values: Value,
    pub(super) vectors: Option<Pair>,
}

/// The input divided by its largest magnitude, its adjoint, and whether it is
/// Hermitian within the eigensolver's tolerance.
struct HermitianScaling {
    scale: Value,
    scaled: Pair,
    adjoint: Pair,
    tolerance: Value,
    hermitian: Value,
    one: Value,
}

impl Emitter<'_> {
    fn eigen_epsilon(&self) -> f64 {
        match self.dtype {
            Dtype::F32 => f32::EPSILON as f64,
            Dtype::F64 => f64::EPSILON,
        }
    }

    fn hermitian_scaling(&mut self, input: &Pair, n: u64) -> HermitianScaling {
        let epsilon = self.eigen_epsilon();
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
        HermitianScaling {
            scale,
            scaled,
            adjoint,
            tolerance,
            hermitian,
            one,
        }
    }

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
                .all(|x| matches!(x, Scalar::Real(z) if z == 0.0) || matches!(x, Scalar::Int(0)));
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
                    let eps = e.scalar(epsilon);
                    let (cosine, w, _) =
                        e.jacobi_rotation(&alpha, &beta, (&gamma, None), &eps, &eps, None);
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
        let real = self.batch_rank(&input.0) == 0 && self.known_zero(&input.1);
        // Explicit host batches keep primitive replay, as the solve helpers do.
        if n > 0 && real && !self.host_batch {
            return Ok(self.symmetric_eigen_helper(&input, n, vectors));
        }
        Ok(self.hermitian_eigen(input, n, vectors, real, true).0)
    }

    /// Unbatched real symmetric eigenpairs through the versioned private
    /// helper `__enzyme_linalg_eigh_v1_{dtype}_{target}_n{n}`. Enzyme-JAX
    /// wheels that recognise it differentiate it analytically; any other
    /// consumer inlines and differentiates its body, the Jacobi solver. The
    /// helper returns ascending eigenvalues, eigenvector columns at canonical
    /// phase and one validity flag, so the caller repeats the O(n^2) symmetry
    /// check to keep its own runtime error apart from non-convergence.
    fn symmetric_eigen_helper(&mut self, input: &Pair, n: u64, vectors: bool) -> HermitianEigen {
        let dtype = match self.dtype {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
        };
        let target = match self.target {
            crate::Target::Cpu => "cpu",
            crate::Target::Gpu => "gpu",
        };
        let name = format!("__enzyme_linalg_eigh_v1_{dtype}_{target}_n{n}");
        let matrix = MlirTy::Ranked(vec![Some(n), Some(n)]);
        if !self.solve_helpers.contains_key(&name) {
            let mut helper = self.scratch_emitter();
            let a = Value {
                ssa: "%a".into(),
                ty: matrix.clone(),
                elem: ElemKind::Real,
            };
            let zero = helper.constant_like(0.0, &a);
            let (eigen, valid) = helper.hermitian_eigen((a.clone(), zero), n, true, true, false);
            let valid = valid.expect("a non-empty eigensolve reports its validity");
            let (eigenvectors, _) = eigen.vectors.expect("requested eigenvectors");
            let args = [(a.ssa, a.ty, a.elem)];
            let rets = [&eigen.values, &eigenvectors, &valid];
            let definition = helper.finish_function(&name, &args, &rets, true);
            self.solve_helpers.insert(name.clone(), definition);
        }
        let hermitian = self.hermitian_scaling(input, n).hermitian;
        self.record_runtime_error(&hermitian, 12);
        let vector = MlirTy::Ranked(vec![Some(n)]);
        let matrix_ty = matrix.render(self.dtype, ElemKind::Real);
        let vector_ty = vector.render(self.dtype, ElemKind::Real);
        let result = self.fresh();
        self.push(&format!(
            "{result}:3 = func.call @{name}({}) : ({matrix_ty}) -> ({vector_ty}, {matrix_ty}, tensor<i1>)",
            input.0.ssa,
        ));
        let values = Value {
            ssa: format!("{result}#0"),
            ty: vector,
            elem: ElemKind::Real,
        };
        let valid = Value {
            ssa: format!("{result}#2"),
            ty: MlirTy::Scalar,
            elem: ElemKind::Bool,
        };
        // The helper already includes the symmetry check, which is recorded
        // first, so this reports only non-convergence.
        self.record_runtime_error(&valid, 10);
        let vectors = vectors.then(|| {
            let re = Value {
                ssa: format!("{result}#1"),
                ty: matrix,
                elem: ElemKind::Real,
            };
            let zero = self.constant_like(0.0, &re);
            let nan = self.scalar(f64::NAN);
            let im = self.select(&valid, &zero, &nan);
            (re, im)
        });
        HermitianEigen { values, vectors }
    }

    /// Ascending eigenpairs and, for a non-empty matrix, their validity: the
    /// input is Hermitian within tolerance and the iteration converged. With
    /// `record` the two failures report runtime errors 12 and 10; a helper
    /// body returns the validity to its caller instead. `real` selects the
    /// unbatched real loop.
    fn hermitian_eigen(
        &mut self,
        input: Pair,
        n: u64,
        vectors: bool,
        real: bool,
        record: bool,
    ) -> (HermitianEigen, Option<Value>) {
        let axes = self.axes_of(&input.0);
        let batch = self.batch_rank(&input.0);
        let rows = self.axis_indices(&input.0, batch);
        let cols = self.axis_indices(&input.0, batch + 1);
        let diagonal = self.compare("EQ", &rows, &cols);
        let identity = self.convert(&diagonal, ElemKind::Real);
        let zero_matrix = self.constant_like(0.0, &input.0);
        if n == 0 {
            let values = self.diagonal_axes(&input.0, batch, batch + 1);
            let eigen = HermitianEigen {
                values,
                vectors: vectors.then_some((identity, zero_matrix)),
            };
            return (eigen, None);
        }
        let epsilon = self.eigen_epsilon();
        let HermitianScaling {
            scale,
            scaled,
            adjoint,
            tolerance,
            hermitian,
            one,
        } = self.hermitian_scaling(&input, n);
        if record {
            self.record_runtime_error(&hermitian, 12);
        }
        let half = self.scalar(0.5);
        let re = self.add(&scaled.0, &adjoint.0);
        let im = self.add(&scaled.1, &adjoint.1);
        let re = self.mul(&half, &re);
        let im = self.mul(&half, &im);
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
                    let eps = e.scalar(epsilon);
                    let ordered = e.compare("LT", &p, &q);
                    let (cosine, wr, wi) = e.jacobi_rotation(
                        &alpha,
                        &beta,
                        (&gamma.0, Some(&gamma.1)),
                        &eps,
                        &eps,
                        Some(&ordered),
                    );
                    // svd_rotate scales its phase by the sine; w = c t already
                    // carries it.
                    let w = (wr, wi.expect("complex coupling"));
                    let right = e.svd_rotate(&a, &p, &q, &cosine, &one, &w);
                    let adjoint = e.hermitian_adjoint(&right);
                    let both = e.svd_rotate(&adjoint, &p, &q, &cosine, &one, &w);
                    let both = e.hermitian_adjoint(&both);
                    let mut next = vec![both.0, both.1];
                    if vectors {
                        let v = e.svd_rotate(
                            &(state[2].clone(), state[3].clone()),
                            &p,
                            &q,
                            &cosine,
                            &one,
                            &w,
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
        if record {
            self.record_runtime_error(&converged, 10);
        }
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
        (HermitianEigen { values, vectors }, Some(valid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatppl_core::Idx;

    fn matrix(ssa: &str, batch: usize, n: u64) -> Value {
        let mut dims = vec![Some(3); batch];
        dims.extend([Some(n), Some(n)]);
        Value {
            ssa: ssa.into(),
            ty: MlirTy::Ranked(dims),
            elem: ElemKind::Real,
        }
    }

    /// Enzyme-JAX recognises `__enzyme_linalg_eigh_v1_{dtype}_{target}_n{n}`
    /// only as a private function of one non-empty square real matrix that
    /// returns the eigenvalues, the eigenvector columns and an `i1` validity.
    #[test]
    fn unbatched_real_eigh_calls_the_recognised_helper() {
        use crate::Target::{Cpu, Gpu};
        let module = Module::new();
        let id = NodeId::from_usize(0);
        for (dtype, target, n, case, helper) in [
            (Dtype::F32, Cpu, 5, "eigh", true),
            (Dtype::F64, Cpu, 1, "eigvalsh", true),
            (Dtype::F64, Gpu, 3, "eigh", true),
            (Dtype::F64, Cpu, 0, "eigh", false),
            (Dtype::F64, Cpu, 3, "host batch", false),
            (Dtype::F64, Cpu, 3, "batched", false),
            (Dtype::F64, Cpu, 3, "complex", false),
        ] {
            let vectors = case != "eigvalsh";
            let host_batch = case == "host batch";
            let batch = usize::from(case == "batched");
            let complex = case == "complex";
            let mut emitter = Emitter::with_options(
                &module,
                &crate::EmitOptions {
                    dtype,
                    target,
                    ..Default::default()
                },
            );
            if host_batch {
                emitter.lower_for_host_batch();
            }
            let re = matrix("%arg0", batch, n);
            let im = matrix("%arg1", batch, n);
            for value in [&re, &im] {
                emitter.remember_axes(
                    &value.ssa,
                    Axes {
                        batch,
                        layers: vec![2],
                    },
                );
            }
            let imaginary = if complex {
                im.clone()
            } else {
                emitter.constant_like(0.0, &re)
            };
            // Two solves of the same shape share one definition.
            let first = emitter
                .matrix_hermitian(id, (re.clone(), imaginary.clone()), vectors)
                .unwrap();
            let second = emitter
                .matrix_hermitian(id, (re.clone(), imaginary), vectors)
                .unwrap();
            let codes = emitter.runtime_codes();
            let mut rets = vec![&first.values, &second.values];
            if let Some((vectors, _)) = &first.vectors {
                rets.push(vectors);
            }
            let args = [
                (re.ssa.clone(), re.ty.clone(), re.elem),
                (im.ssa.clone(), im.ty.clone(), im.elem),
            ];
            let ir = emitter.finish("main", &args, &rets);
            let t = match dtype {
                Dtype::F32 => "f32",
                Dtype::F64 => "f64",
            };
            let case = format!("{t} {target:?} n={n} {case}:\n{ir}");
            if !helper {
                assert!(!ir.contains("__enzyme_linalg_eigh"), "{case}");
                continue;
            }
            let name = format!("{t}_{}_n{n}", format!("{target:?}").to_lowercase());
            let signature = format!(
                "  func.func private @__enzyme_linalg_eigh_v1_{name}(%a: tensor<{n}x{n}x{t}>) \
                 -> (tensor<{n}x{t}>, tensor<{n}x{n}x{t}>, tensor<i1>) {{"
            );
            assert_eq!(
                ir.lines().filter(|line| *line == signature).count(),
                1,
                "{case}"
            );
            assert_eq!(
                ir.matches(&format!("func.call @__enzyme_linalg_eigh_v1_{name}("))
                    .count(),
                2,
                "{case}"
            );
            // The caller keeps a non-Hermitian input (12) apart from
            // non-convergence (10).
            assert!(
                codes.contains(&12) && codes.contains(&10),
                "{codes:?} {case}"
            );
        }
    }
}
