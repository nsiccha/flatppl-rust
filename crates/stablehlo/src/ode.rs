//! `odesolve(f, y0, t0, ts[, p]; rtol, atol, max_steps)` (dogfood-fork
//! extension, not yet in flatppl-design): the solution of `dy/dt = f(t, y[, p])`,
//! `y(t0) = y0`, at every time in `ts`.
//!
//! The lowering is an adaptive explicit Dormand–Prince 5(4) integrator with
//! first-same-as-last stages, using the error control of Boost.Odeint's
//! `controlled_runge_kutta`, which Stan's `ode_rk45` uses:
//!
//! - error `max_i |err_i| / (atol + rtol (|y_i| + h |f_i|))` (infinity norm,
//!   `default_error_checker` with `a_x = a_dxdt = 1`);
//! - a rejected step shrinks `h` by `max(0.9 err^(-1/3), 0.2)`, and an accepted
//!   step with `err < 0.5` grows it by `0.9 max(err, 5^-5)^(-1/5)`
//!   (`default_step_adjuster`, error order 4, stepper order 5);
//! - the first trial step is 0.1, the step size Stan passes;
//! - `max_steps` bounds the accepted steps between consecutive output times
//!   (Stan's `max_num_steps`), and 500 consecutive rejections end the interval
//!   (Odeint's `failed_step_checker`).
//!
//! Unlike Stan's dense-output integration, a step is clamped to end exactly at
//! the next output time, so the step sequence and the result differ from Stan's
//! within the requested tolerance. An interval that does not reach its output
//! time (step budget exhausted, a non-finite derivative, or a decreasing `ts`)
//! yields NaN states rather than a silently wrong value.
//!
//! One `stablehlo.while` per output time is nested in a counted loop over `ts`.
//! The inner loop's trip count depends on the inputs, so it is not a counted
//! loop: Enzyme-JAX's reverse mode refuses it today ("WhileOp does not have
//! known iteration count for cache removal"; KB decision `1jb3rn9`).

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
const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

const DEFAULT_RTOL: f64 = 1e-6;
const DEFAULT_ATOL: f64 = 1e-6;
const DEFAULT_MAX_STEPS: i64 = 1_000_000;
const INITIAL_STEP: f64 = 0.1;
const MAX_CONSECUTIVE_REJECTIONS: i64 = 500;

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
        let k1 = self.ode_rhs(id, &rhs, &t0_value, &y0_value)?;
        let h0 = self.scalar(INITIAL_STEP);
        let counter = self.int_value_const(0);
        // Outer carry: index, t, y, h, k1 (= f(t, y)), buffer.
        let inits = vec![counter, t0_value.clone(), y0_value.clone(), h0, k1, buffer];
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
                let target = e.scan_slice(&ts_value, &index);
                let (t, y, h, k1) = e.ode_interval(
                    id,
                    &rhs,
                    &state_ty,
                    &target,
                    [&args[1], &args[2], &args[3], &args[4]],
                    &rtol,
                    &atol,
                    &max_steps,
                )?;
                // Only an interval that ended exactly at its output time is a
                // solution; anything else is reported as NaN.
                let reached = e.compare("EQ", &t, &target);
                let nan = e.scalar(f64::NAN);
                let nan = e.broadcast_in_dim(&nan, &[], state_ty.clone());
                let out = e.select(&reached, &y, &nan);
                let buffer = e.scan_update(&args[5], &out, &index);
                let one = e.int_value_const(1);
                let next = e.add(&index, &one);
                Ok(vec![next, t, y, h, k1, buffer])
            },
        )?;
        Ok(self.with_typed_axes(id, results[5].clone()))
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

    /// Integrate from the carried `t` to `target`. Returns the final time,
    /// state, proposed step and FSAL derivative.
    #[allow(clippy::too_many_arguments)]
    fn ode_interval(
        &mut self,
        id: NodeId,
        rhs: &Rhs,
        state_ty: &MlirTy,
        target: &Value,
        [t, y, h, k1]: [&Value; 4],
        rtol: &Value,
        atol: &Value,
        max_steps: &Value,
    ) -> Result<(Value, Value, Value, Value), EmitError> {
        let accepted = self.int_value_const(0);
        let rejected = self.int_value_const(0);
        let inits = vec![
            t.clone(),
            y.clone(),
            h.clone(),
            k1.clone(),
            accepted,
            rejected,
        ];
        let types = inits
            .iter()
            .map(|v| v.ty.render(self.dtype, v.elem))
            .collect::<Vec<_>>();
        let target = target.clone();
        let cond_target = target.clone();
        let max_steps = max_steps.clone();
        let results = self.try_while_loop(
            &inits,
            &types,
            |e, args| {
                let open = e.compare("LT", &args[0], &cond_target);
                let budget = e.compare("LT", &args[4], &max_steps);
                let limit = e.int_value_const(MAX_CONSECUTIVE_REJECTIONS);
                let patience = e.compare("LT", &args[5], &limit);
                let go = e.and(&open, &budget);
                e.and(&go, &patience)
            },
            |e, args| {
                let (t, y, h, k1) = (&args[0], &args[1], &args[2], &args[3]);
                let remaining = e.sub(&target, t);
                let clamped = e.compare("LT", &remaining, h);
                let step = e.min(h, &remaining);
                // Stages k2..k7.
                let mut stages = vec![k1.clone()];
                for (row, (&c, coefficients)) in C.iter().zip(A.iter()).enumerate() {
                    let mut increment: Option<Value> = None;
                    for (j, &a) in coefficients.iter().enumerate() {
                        if a == 0.0 {
                            continue;
                        }
                        let a = e.scalar(a);
                        let term = e.mul(&a, &stages[j]);
                        increment = Some(match increment {
                            Some(sum) => e.add(&sum, &term),
                            None => term,
                        });
                    }
                    let increment = increment.expect("each stage row has a nonzero coefficient");
                    let increment = e.mul(&step, &increment);
                    let y_stage = e.add(y, &increment);
                    if row == 5 {
                        // The last row is the 5th-order solution itself.
                        let t_end = e.add(t, &step);
                        let t_end = e.select(&clamped, &target, &t_end);
                        let k7 = e.ode_rhs(id, rhs, &t_end, &y_stage)?;
                        stages.push(k7);
                        stages.push(y_stage);
                        stages.push(t_end);
                        break;
                    }
                    let c = e.scalar(c);
                    let dt = e.mul(&c, &step);
                    let t_stage = e.add(t, &dt);
                    let k = e.ode_rhs(id, rhs, &t_stage, &y_stage)?;
                    stages.push(k);
                }
                let t_end = stages.pop().expect("end time");
                let y_new = stages.pop().expect("5th-order state");
                // stages = k1..k7.
                let mut error: Option<Value> = None;
                for (&w, k) in E.iter().zip(&stages) {
                    if w == 0.0 {
                        continue;
                    }
                    let w = e.scalar(w);
                    let term = e.mul(&w, k);
                    error = Some(match error {
                        Some(sum) => e.add(&sum, &term),
                        None => term,
                    });
                }
                let error = error.expect("nonzero error weights");
                let error = e.mul(&step, &error);
                let error = e.abs(&error);
                let y_abs = e.abs(y);
                let k1_abs = e.abs(k1);
                let h_abs = e.abs(&step);
                let slope = e.mul(&h_abs, &k1_abs);
                let scale = e.add(&y_abs, &slope);
                let scale = e.mul(rtol, &scale);
                let scale = e.add(atol, &scale);
                let ratio = {
                    let r = e.div(&error, &scale);
                    let (r, _) = e.broadcast_pair(&r, y);
                    r
                };
                let err = e.reduce_max(&ratio);
                let one = e.scalar(1.0);
                let accept = e.compare("LE", &err, &one);
                // Step adjustment (Boost default_step_adjuster).
                let shrink = {
                    let p = e.scalar(-1.0 / 3.0);
                    let f = e.pow(&err, &p);
                    let nine = e.scalar(0.9);
                    let f = e.mul(&nine, &f);
                    let floor = e.scalar(0.2);
                    let f = e.max(&f, &floor);
                    e.mul(&step, &f)
                };
                let grow = {
                    let floor = e.scalar(5f64.powi(-5));
                    let x = e.max(&err, &floor);
                    let p = e.scalar(-1.0 / 5.0);
                    let f = e.pow(&x, &p);
                    let nine = e.scalar(0.9);
                    let f = e.mul(&nine, &f);
                    let half = e.scalar(0.5);
                    let small = e.compare("LT", &err, &half);
                    let grown = e.mul(&step, &f);
                    let grown = e.select(&small, &grown, &step);
                    // A step shortened to land on the output time keeps the
                    // larger proposal for the next interval.
                    let kept = e.max(h, &grown);
                    e.select(&clamped, &kept, &grown)
                };
                let h_next = e.select(&accept, &grow, &shrink);
                let t_next = e.select(&accept, &t_end, t);
                let y_next = e.select(&accept, &y_new, y);
                let k7 = stages[6].clone();
                let k1_next = e.select(&accept, &k7, k1);
                let one_int = e.int_value_const(1);
                let zero_int = e.int_value_const(0);
                let accepted = e.add(&args[4], &one_int);
                let accepted = e.select(&accept, &accepted, &args[4]);
                let rejected = e.add(&args[5], &one_int);
                let rejected = e.select(&accept, &zero_int, &rejected);
                let _ = state_ty;
                Ok(vec![t_next, y_next, h_next, k1_next, accepted, rejected])
            },
        )?;
        Ok((
            results[0].clone(),
            results[1].clone(),
            results[2].clone(),
            results[3].clone(),
        ))
    }
}
