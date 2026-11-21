//! `odesolve(f, y0, t0, ts[, p]; rtol, atol, max_steps)` (dogfood-fork
//! extension, not yet in flatppl-design): the solution of `dy/dt = f(t, y[, p])`,
//! `y(t0) = y0`, at every time in `ts`.
//!
//! The lowering reproduces Stan's `ode_rk45` (Stan Math 5 with Boost 1.87, as
//! bundled with BridgeStan 2.9.0): Boost.Odeint's dense-output `integrate_times`
//! driving `controlled_runge_kutta<runge_kutta_dopri5>` with
//! `make_dense_output(atol, rtol, ...)`, a first step of 0.1 and the output
//! times preceded by `t0`. Concretely, following the Boost sources:
//!
//! - steps run freely past output times; each output time `t_out` is read off
//!   the dopri5 dense-output interpolant of the step that covers it, as soon as
//!   `t_out - t <= eps` (`less_eq_with_sign`, `eps` the float type's epsilon);
//! - a step whose end would pass the last output time by more than `eps` is
//!   replaced by one ending exactly there;
//! - error `max_i |err_i| / (atol + rtol (|y_i| + |h| |f_i|))` (infinity norm,
//!   `default_error_checker` with `a_x = a_dxdt = 1`); a step is rejected when
//!   it exceeds 1 and `h` shrinks by `max(0.9 err^(-1/3), 0.2)`; an accepted
//!   step with `err < 0.5` grows `h` by `0.9 max(err, 5^-5)^(-1/5)`;
//! - stages, solution and interpolant are summed in Boost's order,
//!   `y + (h a_1) k_1 + (h a_2) k_2 + ...`;
//! - `max_steps` bounds the accepted steps between consecutive output times
//!   (`max_step_checker`, reset at every observation), and a step attempt that
//!   fails 501 times in a row stops (`failed_step_checker`).
//!
//! Where Stan throws (step budget exhausted, step size not found) the affected
//! output times and all later ones are NaN rather than a silently wrong value.
//! One difference remains: Boost accepts a step whose error estimate is NaN,
//! this lowering rejects it; either way the solution becomes NaN.
//!
//! One `stablehlo.while` of steps per output time is nested in a counted loop
//! over `ts`. The step loop's trip count depends on the inputs, so it is not a
//! counted loop. Enzyme-JAX's reverse mode refuses it today ("WhileOp does not
//! have known iteration count for cache removal"), while its forward mode
//! differentiates it. That is a limitation to lift in Enzyme-JAX; the loop is
//! deliberately not reshaped into a fixed trip count around it.

use super::sequences::Components;
use super::*;

/// Dormand–Prince 5(4) tableau: nodes, stage coefficients, and the error
/// weights `b - b*` (the 5th-order weights are the last stage row).
const C: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
const A: [&[f64]; 6] = [
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    &[
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ],
    &[
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];
/// Error weights `b - b*`, computed as Boost computes them (`c_i - b*_i` in
/// floating point), so step decisions round as Stan's do.
const E: [f64; 7] = [
    35.0 / 384.0 - 5179.0 / 57600.0,
    0.0,
    500.0 / 1113.0 - 7571.0 / 16695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2187.0 / 6784.0 - -92097.0 / 339200.0,
    11.0 / 84.0 - 187.0 / 2100.0,
    -1.0 / 40.0,
];

const DEFAULT_RTOL: f64 = 1e-6;
const DEFAULT_ATOL: f64 = 1e-6;
const DEFAULT_MAX_STEPS: i64 = 1_000_000;
const INITIAL_STEP: f64 = 0.1;
/// `failed_step_checker` throws on the 501st consecutive failure.
const MAX_CONSECUTIVE_REJECTIONS: i64 = 501;

/// The right-hand side `f` with its callable body, inputs and the constant
/// parameter `p` (if any).
struct Rhs {
    body: NodeId,
    entries: Vec<(Symbol, Ref)>,
    parameter: Option<(Components, Type)>,
    state_type: Type,
}

impl Emitter<'_> {
    pub(super) fn lower_odesolve(
        &mut self,
        id: NodeId,
        args: &[NodeId],
    ) -> Result<Value, EmitError> {
        if !self.broadcast_frame.is_empty() || !self.call_frame.is_empty() {
            return Err(EmitError::at(
                id,
                "odesolve: a broadcast or mapped odesolve is not lowered to StableHLO",
            ));
        }
        let [function, y0, t0, ts, rest @ ..] = args else {
            return Err(EmitError::at(
                id,
                "odesolve(f, y0, t0, ts[, p]): invalid argument count",
            ));
        };
        if rest.len() > 1 {
            return Err(EmitError::at(id, "odesolve takes one final parameter `p`"));
        }
        let (body, entries) = self.callable_parts(*function)?;
        if entries.len() != 2 + rest.len() {
            return Err(EmitError::at(
                *function,
                "odesolve: `f` must take the time, the state and the optional parameter `p`",
            ));
        }
        let mut rtol = None;
        let mut atol = None;
        let mut max_steps = None;
        let named = match self.m.node(id) {
            Node::Call(call) => call.named.to_vec(),
            _ => Vec::new(),
        };
        for arg in named {
            let slot = match self.m.resolve(arg.name) {
                "rtol" => &mut rtol,
                "atol" => &mut atol,
                "max_steps" => &mut max_steps,
                other => {
                    return Err(EmitError::at(
                        arg.value,
                        format!("odesolve has no keyword input `{other}`"),
                    ));
                }
            };
            *slot = Some(arg.value);
        }

        let y0_value = self.lower_node(*y0)?;
        let y0_value = self.convert(&y0_value, ElemKind::Real);
        let d = match shape(&y0_value.ty) {
            [Some(d)] if self.batch_rank(&y0_value) == 0 => *d,
            _ => {
                return Err(EmitError::at(
                    *y0,
                    "odesolve: the initial state must be a vector of static length",
                ));
            }
        };
        let state_ty = MlirTy::Ranked(vec![Some(d)]);
        let t0_value = self.lower_node(*t0)?;
        let t0_value = self.convert(&t0_value, ElemKind::Real);
        if t0_value.ty != MlirTy::Scalar {
            return Err(EmitError::at(
                *t0,
                "odesolve: the initial time must be a scalar",
            ));
        }
        let ts_value = self.lower_node(*ts)?;
        let ts_value = self.convert(&ts_value, ElemKind::Real);
        let n = match shape(&ts_value.ty) {
            [Some(n)] if self.batch_rank(&ts_value) == 0 => *n,
            _ => {
                return Err(EmitError::at(
                    *ts,
                    "odesolve: the output times must be a vector of static length",
                ));
            }
        };
        let real_option = |e: &mut Self,
                           node: Option<NodeId>,
                           default: f64|
         -> Result<Value, EmitError> {
            match node {
                Some(node) => {
                    let value = e.lower_node(node)?;
                    if value.ty != MlirTy::Scalar {
                        return Err(EmitError::at(node, "odesolve: tolerances must be scalars"));
                    }
                    Ok(e.convert(&value, ElemKind::Real))
                }
                None => Ok(e.scalar(default)),
            }
        };
        let rtol = real_option(self, rtol, DEFAULT_RTOL)?;
        let atol = real_option(self, atol, DEFAULT_ATOL)?;
        let max_steps = match max_steps {
            Some(node) => {
                let value = self.lower_node(node)?;
                if value.ty != MlirTy::Scalar || value.elem != ElemKind::Int {
                    return Err(EmitError::at(
                        node,
                        "odesolve: `max_steps` must be an integer",
                    ));
                }
                value
            }
            None => self.int_value_const(DEFAULT_MAX_STEPS),
        };
        let parameter =
            rest.first()
                .map(|&node| {
                    let ty = self.type_of(node).cloned().ok_or_else(|| {
                        EmitError::at(node, "odesolve: unresolved parameter type")
                    })?;
                    Ok::<_, EmitError>((self.lower_components(node)?, ty))
                })
                .transpose()?;
        let rhs = Rhs {
            body,
            entries,
            parameter,
            state_type: Type::Array {
                shape: Box::new([flatppl_core::Dim::Static(
                    u32::try_from(d)
                        .map_err(|_| EmitError::at(*y0, "odesolve: the state is too long"))?,
                )]),
                elem: Box::new(Type::Scalar(flatppl_core::ScalarType::Real)),
            },
        };

        // The output buffer holds one state per output time.
        let zero = self.scalar(0.0);
        let buffer = self.broadcast_in_dim(&zero, &[], MlirTy::Ranked(vec![Some(n), Some(d)]));
        if n == 0 {
            return Ok(self.with_typed_axes(id, buffer));
        }
        let last_index = self.int_value_const(n as i64 - 1);
        let last_time = self.scan_slice(&ts_value, &last_index);
        let eps = self.scalar(match self.dtype {
            Dtype::F32 => f64::from(f32::EPSILON),
            Dtype::F64 => f64::EPSILON,
        });
        let k0 = self.ode_rhs(id, &rhs, &t0_value, &y0_value)?;
        let h0 = self.scalar(INITIAL_STEP);
        let counter = self.int_value_const(0);
        let not_failed = self.bool_value_const(false);
        let zeros = self.broadcast_in_dim(&zero, &[], state_ty.clone());
        // The current point (t, y, f), the proposed step h, the last accepted
        // step's start (t, y, f) and its stages k3..k6 for the interpolant.
        let step = Step {
            t: t0_value.clone(),
            y: y0_value.clone(),
            k: k0.clone(),
            h: h0,
            t_old: t0_value,
            y_old: y0_value,
            k_old: k0,
            stages: [zeros.clone(), zeros.clone(), zeros.clone(), zeros],
        };
        let mut inits = vec![counter, not_failed];
        inits.extend(step.values());
        inits.push(buffer);
        let types = inits
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let results = self.try_while_loop(
            &inits,
            &types,
            |e, args| {
                let limit = e.int_value_const(n as i64);
                e.compare("LT", &args[0], &limit)
            },
            |e, args| {
                let index = args[0].clone();
                let failed = args[1].clone();
                let carried = Step::from_values(&args[2..2 + Step::LEN]);
                let target = e.scan_slice(&ts_value, &index);
                let limits = Limits {
                    target: &target,
                    last: &last_time,
                    eps: &eps,
                    rtol: &rtol,
                    atol: &atol,
                    max_steps: &max_steps,
                    failed: &failed,
                };
                let step = e.ode_steps(id, &rhs, carried, &limits)?;
                // `less_eq_with_sign(t_out, t, h)`: the output time is covered.
                let ahead = e.sub(&target, &step.t);
                let covered = e.compare("LE", &ahead, &eps);
                let ok = e.not(&failed);
                let ok = e.and(&ok, &covered);
                let value = e.ode_interpolate(&step, &target);
                let nan = e.scalar(f64::NAN);
                let nan = e.broadcast_in_dim(&nan, &[], state_ty.clone());
                let out = e.select(&ok, &value, &nan);
                let buffer = e.scan_update(&args[2 + Step::LEN], &out, &index);
                let one = e.int_value_const(1);
                let mut next = vec![e.add(&index, &one), e.not(&ok)];
                next.extend(step.values());
                next.push(buffer);
                Ok(next)
            },
        )?;
        Ok(self.with_typed_axes(id, results[2 + Step::LEN].clone()))
    }

    /// `f(t, y[, p])` as a real vector of the state's length.
    fn ode_rhs(&mut self, id: NodeId, rhs: &Rhs, t: &Value, y: &Value) -> Result<Value, EmitError> {
        let mut arguments = vec![
            Components::flat(None, vec![t.clone()]),
            Components::flat(None, vec![y.clone()]),
        ];
        let mut types = vec![
            Type::Scalar(flatppl_core::ScalarType::Real),
            rhs.state_type.clone(),
        ];
        if let Some((value, ty)) = &rhs.parameter {
            arguments.push(value.clone());
            types.push(ty.clone());
        }
        let template = Components::flat(None, vec![y.clone()]);
        let value = self
            .call_components(rhs.body, &rhs.entries, &arguments, &types)?
            .align(&template, id)?
            .tensor(id)?;
        let value = self.convert(&value, ElemKind::Real);
        if value.ty != y.ty {
            let (value, _) = self.broadcast_pair(&value, y);
            if value.ty != y.ty {
                return Err(EmitError::at(
                    id,
                    "odesolve: `f` must return dy/dt with the shape of the state",
                ));
            }
            return Ok(value);
        }
        Ok(value)
    }

    /// Boost's dopri5 dense output at `t_out` within the last accepted step:
    /// `y_old + (h b1(θ)) k_old + (h b3(θ)) k3 + ... + (h b7(θ)) k_new` with
    /// `θ = (t_out - t_old) / h`. Before any step (`t == t_old`) it is `y`.
    fn ode_interpolate(&mut self, step: &Step, t_out: &Value) -> Value {
        let h = self.sub(&step.t, &step.t_old);
        let offset = self.sub(t_out, &step.t_old);
        let theta = self.div(&offset, &h);
        let c = |e: &mut Self, x: f64| e.scalar(x);
        let one = c(self, 1.0);
        let theta_m_1 = self.sub(&theta, &one);
        let theta_sq = self.mul(&theta, &theta);
        // A = θ²(3 - 2θ), B = θ²(θ-1), C = θ²(θ-1)², D = θ(θ-1)².
        let a = {
            let two = c(self, 2.0);
            let three = c(self, 3.0);
            let two_theta = self.mul(&two, &theta);
            let f = self.sub(&three, &two_theta);
            self.mul(&theta_sq, &f)
        };
        let b = self.mul(&theta_sq, &theta_m_1);
        let cc = self.mul(&b, &theta_m_1);
        let dd = {
            let t = self.mul(&theta, &theta_m_1);
            self.mul(&t, &theta_m_1)
        };
        // X_i = scale * (p - q θ) / r, exactly as Boost writes them.
        let x = |e: &mut Self, scale: f64, p: f64, q: f64, r: f64| {
            let p = e.scalar(p);
            let q = e.scalar(q);
            let qt = e.mul(&q, &theta);
            let diff = e.sub(&p, &qt);
            let scale = e.scalar(scale);
            let num = e.mul(&scale, &diff);
            let r = e.scalar(r);
            e.div(&num, &r)
        };
        let x1 = x(self, 5.0, 2558722523.0, 31403016.0, 11282082432.0);
        let x3 = x(self, 100.0, 882725551.0, 15701508.0, 32700410799.0);
        let x4 = x(self, 25.0, 443332067.0, 31403016.0, 1880347072.0);
        let x5 = x(self, 32805.0, 23143187.0, 3489224.0, 199316789632.0);
        let x6 = x(self, 55.0, 29972135.0, 7076736.0, 822651844.0);
        let x7 = x(self, 10.0, 7414447.0, 829305.0, 29380423.0);
        let weight = |e: &mut Self, b: f64, sign: f64, xi: &Value| {
            let b = e.scalar(b);
            let ab = e.mul(&a, &b);
            let cx = e.mul(&cc, xi);
            if sign > 0.0 {
                e.add(&ab, &cx)
            } else {
                e.sub(&ab, &cx)
            }
        };
        let b1 = {
            let w = weight(self, A[5][0], -1.0, &x1);
            self.add(&w, &dd)
        };
        let b3 = weight(self, A[5][2], 1.0, &x3);
        let b4 = weight(self, A[5][3], -1.0, &x4);
        let b5 = weight(self, A[5][4], 1.0, &x5);
        let b6 = weight(self, A[5][5], -1.0, &x6);
        let b7 = {
            let cx = self.mul(&cc, &x7);
            self.add(&b, &cx)
        };
        let mut value = step.y_old.clone();
        for (w, k) in [
            (b1, &step.k_old),
            (b3, &step.stages[0]),
            (b4, &step.stages[1]),
            (b5, &step.stages[2]),
            (b6, &step.stages[3]),
            (b7, &step.k),
        ] {
            let hw = self.mul(&h, &w);
            let term = self.mul(&hw, k);
            value = self.add(&value, &term);
        }
        let fresh = self.compare("EQ", &step.t, &step.t_old);
        self.select(&fresh, &step.y, &value)
    }

    /// Accepted steps until the output time is covered (Boost's inner
    /// `less_eq_with_sign` test), the step budget is spent or a step size is
    /// not found.
    fn ode_steps(
        &mut self,
        id: NodeId,
        rhs: &Rhs,
        step: Step,
        limits: &Limits<'_>,
    ) -> Result<Step, EmitError> {
        let accepted = self.int_value_const(0);
        let rejected = self.int_value_const(0);
        let mut inits = step.values();
        inits.extend([accepted, rejected]);
        let types = inits
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let results = self.try_while_loop(
            &inits,
            &types,
            |e, args| {
                let ahead = e.sub(limits.target, &args[0]);
                let open = e.compare("GT", &ahead, limits.eps);
                let budget = e.compare("LT", &args[Step::LEN], limits.max_steps);
                let cap = e.int_value_const(MAX_CONSECUTIVE_REJECTIONS);
                let patience = e.compare("LT", &args[Step::LEN + 1], &cap);
                let alive = e.not(limits.failed);
                let go = e.and(&open, &budget);
                let go = e.and(&go, &patience);
                e.and(&go, &alive)
            },
            |e, args| {
                let step = Step::from_values(&args[..Step::LEN]);
                let (t, y, k, h) = (&step.t, &step.y, &step.k, &step.h);
                // A step that would pass the last output time ends on it.
                let end = e.add(t, h);
                let past = e.sub(&end, limits.last);
                let inside = e.compare("LE", &past, limits.eps);
                let to_last = e.sub(limits.last, t);
                let h = e.select(&inside, h, &to_last);
                // Stages k2..k6, then the 5th-order solution and k7 = f(t+h, y5).
                let mut ks = vec![k.clone()];
                for (row, (&node, coefficients)) in C.iter().zip(A.iter()).enumerate() {
                    let mut x = y.clone();
                    for (j, &a) in coefficients.iter().enumerate() {
                        if row == 5 && j == 1 {
                            continue; // b2 = 0 in the solution row.
                        }
                        let a = e.scalar(a);
                        let ha = e.mul(&h, &a);
                        let term = e.mul(&ha, &ks[j]);
                        x = e.add(&x, &term);
                    }
                    if row == 5 {
                        let t_new = e.add(t, &h);
                        let k7 = e.ode_rhs(id, rhs, &t_new, &x)?;
                        ks.push(k7);
                        ks.push(x);
                        ks.push(t_new);
                        break;
                    }
                    let c = e.scalar(node);
                    let hc = e.mul(&h, &c);
                    let t_stage = e.add(t, &hc);
                    let kn = e.ode_rhs(id, rhs, &t_stage, &x)?;
                    ks.push(kn);
                }
                let t_new = ks.pop().expect("end time");
                let y_new = ks.pop().expect("5th-order state");
                // ks = k1..k7. Error estimate in Boost's order.
                let mut error: Option<Value> = None;
                for (&w, kj) in E.iter().zip(&ks) {
                    if w == 0.0 {
                        continue;
                    }
                    let w = e.scalar(w);
                    let hw = e.mul(&h, &w);
                    let term = e.mul(&hw, kj);
                    error = Some(match error {
                        Some(sum) => e.add(&sum, &term),
                        None => term,
                    });
                }
                let error = error.expect("nonzero error weights");
                let error = e.abs(&error);
                let y_abs = e.abs(y);
                let k_abs = e.abs(k);
                let h_abs = e.abs(&h);
                let slope = e.mul(&h_abs, &k_abs);
                let scale = e.add(&y_abs, &slope);
                let scale = e.mul(limits.rtol, &scale);
                let scale = e.add(limits.atol, &scale);
                let ratio = e.div(&error, &scale);
                let (ratio, _) = e.broadcast_pair(&ratio, y);
                let err = e.reduce_max(&ratio);
                let one = e.scalar(1.0);
                let accept = e.compare("LE", &err, &one);
                let shrink = {
                    let p = e.scalar(-1.0 / 3.0);
                    let f = e.pow(&err, &p);
                    let nine = e.scalar(0.9);
                    let f = e.mul(&nine, &f);
                    let floor = e.scalar(0.2);
                    let f = e.max(&f, &floor);
                    e.mul(&h, &f)
                };
                let grow = {
                    let floor = e.scalar(5f64.powi(-5));
                    let x = e.max(&err, &floor);
                    let p = e.scalar(-1.0 / 5.0);
                    let f = e.pow(&x, &p);
                    let nine = e.scalar(0.9);
                    let f = e.mul(&nine, &f);
                    let grown = e.mul(&h, &f);
                    let half = e.scalar(0.5);
                    let small = e.compare("LT", &err, &half);
                    e.select(&small, &grown, &h)
                };
                let pick = |e: &mut Self, new: &Value, old: &Value| e.select(&accept, new, old);
                let next = Step {
                    t: pick(e, &t_new, t),
                    y: pick(e, &y_new, y),
                    k: pick(e, &ks[6], k),
                    h: pick(e, &grow, &shrink),
                    t_old: pick(e, t, &step.t_old),
                    y_old: pick(e, y, &step.y_old),
                    k_old: pick(e, k, &step.k_old),
                    stages: [
                        pick(e, &ks[2], &step.stages[0]),
                        pick(e, &ks[3], &step.stages[1]),
                        pick(e, &ks[4], &step.stages[2]),
                        pick(e, &ks[5], &step.stages[3]),
                    ],
                };
                let one_int = e.int_value_const(1);
                let zero_int = e.int_value_const(0);
                let accepted = e.add(&args[Step::LEN], &one_int);
                let accepted = e.select(&accept, &accepted, &args[Step::LEN]);
                let rejected = e.add(&args[Step::LEN + 1], &one_int);
                let rejected = e.select(&accept, &zero_int, &rejected);
                let mut values = next.values();
                values.extend([accepted, rejected]);
                Ok(values)
            },
        )?;
        Ok(Step::from_values(&results[..Step::LEN]))
    }
}

/// The integrator state carried between steps and output times.
#[derive(Clone)]
struct Step {
    t: Value,
    y: Value,
    k: Value,
    h: Value,
    t_old: Value,
    y_old: Value,
    k_old: Value,
    /// k3, k4, k5, k6 of the last accepted step.
    stages: [Value; 4],
}

impl Step {
    const LEN: usize = 11;

    fn values(&self) -> Vec<Value> {
        let mut v = vec![
            self.t.clone(),
            self.y.clone(),
            self.k.clone(),
            self.h.clone(),
            self.t_old.clone(),
            self.y_old.clone(),
            self.k_old.clone(),
        ];
        v.extend(self.stages.iter().cloned());
        v
    }

    fn from_values(v: &[Value]) -> Self {
        Step {
            t: v[0].clone(),
            y: v[1].clone(),
            k: v[2].clone(),
            h: v[3].clone(),
            t_old: v[4].clone(),
            y_old: v[5].clone(),
            k_old: v[6].clone(),
            stages: [v[7].clone(), v[8].clone(), v[9].clone(), v[10].clone()],
        }
    }
}

/// Per-output-time constants of the step loop.
struct Limits<'a> {
    target: &'a Value,
    last: &'a Value,
    eps: &'a Value,
    rtol: &'a Value,
    atol: &'a Value,
    max_steps: &'a Value,
    failed: &'a Value,
}
