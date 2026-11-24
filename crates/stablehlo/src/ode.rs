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
//!   step with `err < 0.5` grows `h` by `0.9 max(err, 5^-5)^(-1/5)`. Both
//!   factors are computed on every step and one is selected, so the shrink
//!   factor reads `max(err, 1)`, which is `err` wherever a step is rejected:
//!   at a zero error estimate (a steady state) the unclamped `err^(-1/3)` is
//!   infinite, and reverse mode would turn the unselected branch's zero
//!   cotangent into NaN;
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
//! Derivatives treat the step sizes as fixed. The step-size controller reads
//! the error norm only through [`Emitter::without_derivative`], so the step
//! sizes, step times and accept/reject decisions carry no derivative, and
//! differentiating the solution integrates its forward sensitivities with the
//! Dormand–Prince stages of the steps the value took. Stan does not
//! differentiate its step sizes either. Differentiating through the controller
//! instead adds the derivative of each step size with respect to the
//! parameters, a term the ODE's solution does not have: on Lotka–Volterra at
//! rtol 1e-5, atol 1e-3 it put the gradient up to 44% from the exact one, and
//! at a steady state, where the error estimate is exactly zero, reverse mode
//! returned NaN. Unlike Stan, the sensitivities are not part of the error
//! estimate, so their accuracy follows from the steps the value needs.
//!
//! One `stablehlo.while` of steps per output time is nested in a counted loop
//! over `ts`. The step loop's trip count depends on the inputs, so it is not a
//! counted loop. Enzyme-JAX's reverse mode refuses it today ("WhileOp does not
//! have known iteration count for cache removal"), while its forward mode
//! differentiates it. That is a limitation to lift in Enzyme-JAX; the loop is
//! deliberately not reshaped into a fixed trip count around it.
//!
//! [`crate::EmitOptions::ode_sensitivity_control`] reproduces Stan's two
//! answers instead. Stan's `log_density` takes the value-only steps above,
//! while its gradient call integrates the coupled system `[y, S]`, with `S`
//! the sensitivities of `y` to every non-constant component of `y0` and `p`,
//! and its error norm runs over both. The two differ by up to the tolerance
//! (8e-2 in log density on PosteriorDB `lotka_volterra`), and so does the
//! density downstream of the solution. So a logdensity-mode module lowers its
//! outputs twice: once on the value-only steps and once on the coupled steps,
//! sharing every value that does not depend on an odesolve. Each output is
//! `stop(value) + (coupled - stop(coupled))` ([`Emitter::ode_combine`]): the
//! value of the first pass, which equals it wherever `|value| >= 2^-11` and is
//! within `2^-63` of it elsewhere, with the derivative of the second.
//! Combining the solutions instead would evaluate the density's derivative at
//! the value-only states, about 1e-3 off Stan's gradient on that model.
//!
//! The coupled solve carries `S` (one row per direction) beside `y` and
//! updates it with the same Dormand–Prince stages, whose slopes
//! `J_y S + J_p dp` come from one batched `enzyme.fwddiff` of `f` per stage
//! (`f` is emitted as a private function for it). `S` only steers the step
//! sizes: its inputs are taken through [`Emitter::without_derivative`], so
//! reverse mode never differentiates the Jacobian-vector products. The
//! derivative comes from differentiating the coupled solve's states on those
//! fixed steps, which integrates exactly the sensitivities Stan integrates.
//! Where either pass fails, the output is NaN, as Stan's gradient call throws
//! where its coupled solve does. A component counts as constant when the
//! emitter can read all its elements at emit time (embedded data, literals);
//! anything else is a direction, as Stan treats a `var`. The option needs an
//! Enzyme-JAX build whose `hlo_call` accepts Enzyme's dialect, and other modes
//! refuse it.

use super::constants::element_count;
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

        let solve = Solve {
            t0: &t0_value,
            y0: &y0_value,
            ts: &ts_value,
            n,
            d,
            rtol: &rtol,
            atol: &atol,
            max_steps: &max_steps,
        };
        if self.ode_sensitivity_control && !self.ode_two_pass {
            return Err(EmitError::at(
                id,
                "odesolve sensitivity control is lowered for logdensity-mode modules only",
            ));
        }
        let mut directions = None;
        if self.ode_sensitivity_control && n > 0 {
            let (y0_active, sizes) = self.ode_direction_sizes(id, &rhs, &y0_value)?;
            let width = if y0_active { d } else { 0 } + sizes.iter().flatten().sum::<u64>();
            if width > 0 && self.ode_coupled_pass {
                directions = Some(self.ode_directions(id, &rhs, d, y0_active, &sizes, width)?);
            } else if width > 0 {
                // The value pass: the outputs are lowered again on the coupled steps.
                self.ode_coupled_needed = true;
            }
        }
        let value = self.ode_solve(id, &rhs, &solve, directions.as_ref())?;
        Ok(self.with_typed_axes(id, value))
    }

    /// The combination of one output's two lowerings (module docs): the value
    /// of the value pass with the derivative of the coupled pass.
    pub(crate) fn ode_combine(&mut self, value: &Value, coupled: &Value) -> Value {
        if value.ssa == coupled.ssa || value.elem != ElemKind::Real || value.ty != coupled.ty {
            return value.clone();
        }
        let value = self.without_derivative(value);
        let fixed = self.without_derivative(coupled);
        let derivative = self.sub(coupled, &fixed);
        self.add(&value, &derivative)
    }

    /// The states at every output time. With `directions` the steps are those
    /// of the coupled system `[y, S]` (module docs).
    fn ode_solve(
        &mut self,
        id: NodeId,
        rhs: &Rhs,
        solve: &Solve<'_>,
        directions: Option<&Directions>,
    ) -> Result<Value, EmitError> {
        let (n, d) = (solve.n, solve.d);
        let state_ty = MlirTy::Ranked(vec![Some(d)]);
        // The output buffer holds one state per output time.
        let zero = self.scalar(0.0);
        let buffer = self.broadcast_in_dim(&zero, &[], MlirTy::Ranked(vec![Some(n), Some(d)]));
        if n == 0 {
            return Ok(buffer);
        }
        let last_index = self.int_value_const(n as i64 - 1);
        let last_time = self.scan_slice(solve.ts, &last_index);
        let eps = self.scalar(match self.dtype {
            Dtype::F32 => f64::from(f32::EPSILON),
            Dtype::F64 => f64::EPSILON,
        });
        let k0 = self.ode_rhs(id, rhs, solve.t0, solve.y0)?;
        let h0 = self.scalar(INITIAL_STEP);
        let counter = self.int_value_const(0);
        let not_failed = self.bool_value_const(false);
        let zeros = self.broadcast_in_dim(&zero, &[], state_ty.clone());
        let sensitivities = directions.map(|directions| {
            let slope = self.ode_jvp(directions, rhs, solve.t0, solve.y0, &directions.s0);
            [directions.s0.clone(), slope]
        });
        // The current point (t, y, f), the proposed step h, the last accepted
        // step's start (t, y, f) and its stages k3..k6 for the interpolant.
        let step = Step {
            t: solve.t0.clone(),
            y: solve.y0.clone(),
            k: k0.clone(),
            h: h0,
            t_old: solve.t0.clone(),
            y_old: solve.y0.clone(),
            k_old: k0,
            stages: [zeros.clone(), zeros.clone(), zeros.clone(), zeros],
            sensitivities,
        };
        let len = step.len();
        let coupled = directions.is_some();
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
                let carried = Step::from_values(&args[2..2 + len], coupled);
                let target = e.scan_slice(solve.ts, &index);
                let limits = Limits {
                    target: &target,
                    last: &last_time,
                    eps: &eps,
                    rtol: solve.rtol,
                    atol: solve.atol,
                    max_steps: solve.max_steps,
                    failed: &failed,
                };
                let step = e.ode_steps(id, rhs, carried, &limits, directions)?;
                // `less_eq_with_sign(t_out, t, h)`: the output time is covered.
                let ahead = e.sub(&target, &step.t);
                let covered = e.compare("LE", &ahead, &eps);
                let ok = e.not(&failed);
                let ok = e.and(&ok, &covered);
                let value = e.ode_interpolate(&step, &target);
                let nan = e.scalar(f64::NAN);
                let nan = e.broadcast_in_dim(&nan, &[], state_ty.clone());
                let out = e.select(&ok, &value, &nan);
                let buffer = e.scan_update(&args[2 + len], &out, &index);
                let one = e.int_value_const(1);
                let mut next = vec![e.add(&index, &one), e.not(&ok)];
                next.extend(step.values());
                next.push(buffer);
                Ok(next)
            },
        )?;
        Ok(results[2 + len].clone())
    }

    /// Which of `y0` and `p`'s leaves are directions: every one except those
    /// the emitter can read at emit time, and `p`'s non-real leaves. A
    /// direction leaf's size, `None` for a constant leaf.
    fn ode_direction_sizes(
        &self,
        id: NodeId,
        rhs: &Rhs,
        y0: &Value,
    ) -> Result<(bool, Vec<Option<u64>>), EmitError> {
        let y0_active = !self.fixed_everywhere(y0);
        let mut sizes = Vec::new();
        for leaf in rhs.leaves() {
            let active = leaf.elem == ElemKind::Real && !self.fixed_everywhere(&leaf);
            sizes.push(if active {
                Some(element_count(&leaf.ty).ok_or_else(|| {
                    EmitError::at(id, "odesolve: a parameter of `f` needs a static shape")
                })? as u64)
            } else {
                None
            });
        }
        Ok((y0_active, sizes))
    }

    /// The directions of the coupled solve, `width` of them.
    fn ode_directions(
        &mut self,
        id: NodeId,
        rhs: &Rhs,
        d: u64,
        y0_active: bool,
        sizes: &[Option<u64>],
        width: u64,
    ) -> Result<Directions, EmitError> {
        if let Some(node) = self.ode_captured_input(rhs) {
            return Err(EmitError::at(
                node,
                "odesolve sensitivity control: `f` reads a model input other than through its \
                 arguments; pass it in the parameter `p`",
            ));
        }
        let function = self.ode_rhs_function(id, rhs, d)?;
        let s0 = if y0_active {
            self.unit_rows(width, d, 0)
        } else {
            self.constant(0.0, MlirTy::Ranked(vec![Some(width), Some(d)]))
        };
        let mut offset = if y0_active { d } else { 0 };
        let mut seeds = Vec::with_capacity(sizes.len());
        for (leaf, size) in rhs.leaves().iter().zip(sizes) {
            seeds.push(size.map(|size| {
                let rows = self.unit_rows(width, size, offset);
                offset += size;
                let mut dims = if width == 1 {
                    vec![]
                } else {
                    vec![Some(width)]
                };
                dims.extend(shape(&leaf.ty).iter().copied());
                let ty = if dims.is_empty() {
                    MlirTy::Scalar
                } else {
                    MlirTy::Ranked(dims)
                };
                self.reshape(&rows, ty)
            }));
        }
        Ok(Directions {
            width,
            function,
            s0,
            seeds,
        })
    }

    /// Whether every element of `value` is known at emit time.
    fn fixed_everywhere(&self, value: &Value) -> bool {
        element_count(&value.ty)
            .is_some_and(|count| (0..count).all(|i| self.fixed_scalar_at(value, i).is_some()))
    }

    /// `rows x cols` with ones where `row == offset + col`.
    fn unit_rows(&mut self, rows: u64, cols: u64, offset: u64) -> Value {
        let zero = self.constant(0.0, MlirTy::Ranked(vec![Some(rows), Some(cols)]));
        let row = self.axis_indices(&zero, 0);
        let col = self.axis_indices(&zero, 1);
        let offset = self.int_value_const(offset as i64);
        let col = self.add(&col, &offset);
        let hit = self.compare("EQ", &row, &col);
        self.convert(&hit, ElemKind::Real)
    }

    /// A node through which `f`'s body reads a model input, or a value of an
    /// enclosing function, other than through `f`'s own arguments. Such a
    /// value is not in scope in the private function `fwddiff` needs. The
    /// reification rule already refuses a callable that depends on an
    /// `elementof` outside its inputs; this keeps the private function's
    /// scope checked where the lowering relies on it.
    fn ode_captured_input(&self, rhs: &Rhs) -> Option<NodeId> {
        let inputs = flatppl_core::query::promoted_input_names(self.m);
        let mut own: Vec<Ref> = rhs.entries.iter().map(|(_, r)| *r).collect();
        let mut seen = HashSet::new();
        let mut todo = vec![rhs.body];
        while let Some(node) = todo.pop() {
            if !seen.insert(node) {
                continue;
            }
            match self.m.node(node) {
                Node::Ref(r) if own.contains(r) => {}
                Node::Ref(r) => match r.ns {
                    RefNs::SelfMod if inputs.contains(&r.name) => return Some(node),
                    RefNs::SelfMod => {
                        if let Some(binding) = self.m.binding_by_name(r.name) {
                            todo.push(self.m.binding(binding).rhs);
                        }
                    }
                    RefNs::Local => return Some(node),
                    RefNs::Module(_) => {}
                },
                Node::Call(call) => {
                    // A nested function's own inputs are its arguments.
                    match &call.inputs {
                        Some(Inputs::Spec(entries)) => own.extend(entries.iter().map(|(_, r)| *r)),
                        Some(Inputs::Auto) => {
                            if let Some(entries) = self.m.auto_inputs_of(node) {
                                own.extend(entries.iter().map(|(_, r)| *r));
                            }
                        }
                        None => {}
                    }
                    self.m.for_each_child(node, |child| todo.push(child));
                }
                _ => self.m.for_each_child(node, |child| todo.push(child)),
            }
        }
        None
    }

    /// `f(t, y, p...)` as a private function of `t`, `y` and `p`'s leaves, for
    /// `enzyme.fwddiff`.
    fn ode_rhs_function(&mut self, id: NodeId, rhs: &Rhs, d: u64) -> Result<String, EmitError> {
        let index = self
            .solve_helpers
            .keys()
            .filter(|name| name.starts_with(RHS_FUNCTION))
            .count();
        let name = format!("{RHS_FUNCTION}{index}");
        let mut helper = self.scratch_emitter();
        let t = Value {
            ssa: "%t".into(),
            ty: MlirTy::Scalar,
            elem: ElemKind::Real,
        };
        let y = Value {
            ssa: "%y".into(),
            ty: MlirTy::Ranked(vec![Some(d)]),
            elem: ElemKind::Real,
        };
        let leaves = rhs
            .leaves()
            .iter()
            .enumerate()
            .map(|(i, leaf)| Value {
                ssa: format!("%p{i}"),
                ty: leaf.ty.clone(),
                elem: leaf.elem,
            })
            .collect::<Vec<_>>();
        let local = Rhs {
            body: rhs.body,
            entries: rhs.entries.clone(),
            parameter: rhs
                .parameter
                .as_ref()
                .map(|(components, ty)| (components.with_values(&leaves), ty.clone())),
            state_type: rhs.state_type.clone(),
        };
        let f = helper.ode_rhs(id, &local, &t, &y)?;
        let mut args = vec![(t.ssa, t.ty, t.elem), (y.ssa, y.ty, y.elem)];
        args.extend(
            leaves
                .into_iter()
                .map(|leaf| (leaf.ssa, leaf.ty, leaf.elem)),
        );
        let definition = helper.finish_function(&name, &args, &[&f], true);
        self.solve_helpers.insert(name.clone(), definition);
        Ok(name)
    }

    /// The slopes of the sensitivities at `(t, y)`, one row per direction:
    /// `J_y S + J_p dp` from one batched `enzyme.fwddiff` of `f`. Every input
    /// is taken without its derivative (module docs).
    fn ode_jvp(
        &mut self,
        directions: &Directions,
        rhs: &Rhs,
        t: &Value,
        y: &Value,
        s: &Value,
    ) -> Value {
        const CONST: &str = "#enzyme.activity<enzyme_const>";
        const DUP: &str = "#enzyme.activity<enzyme_dup>";
        let width = directions.width;
        let state = y.ty.clone();
        let y = self.without_derivative(y);
        let s = self.without_derivative(s);
        // One direction is the unbatched form: shadows have the primal types.
        let s = if width == 1 {
            self.reshape(&s, state.clone())
        } else {
            s
        };
        let render = |e: &Self, v: &Value| v.ty.render(e.dtype, v.elem);
        let mut operands = vec![t.ssa.clone(), y.ssa.clone(), s.ssa.clone()];
        let mut types = vec![render(self, t), render(self, &y), render(self, &s)];
        let mut activity = vec![CONST, DUP];
        for (leaf, seed) in rhs.leaves().iter().zip(&directions.seeds) {
            match seed {
                Some(seed) => {
                    let leaf = self.without_derivative(leaf);
                    operands.extend([leaf.ssa.clone(), seed.ssa.clone()]);
                    types.extend([render(self, &leaf), render(self, seed)]);
                    activity.push(DUP);
                }
                None => {
                    operands.push(leaf.ssa.clone());
                    types.push(render(self, leaf));
                    activity.push(CONST);
                }
            }
        }
        let d = shape(&state)[0];
        let result_ty = if width == 1 {
            state
        } else {
            MlirTy::Ranked(vec![Some(width), d])
        };
        let width_attr = if width == 1 {
            String::new()
        } else {
            format!(", width = {width} : i64")
        };
        let ssa = self.fresh();
        self.push(&format!(
            "{ssa} = enzyme.fwddiff @{}({}) <{{activity=[{}], \
             ret_activity=[#enzyme.activity<enzyme_dupnoneed>]{width_attr}}}> : ({}) -> {}",
            directions.function,
            operands.join(", "),
            activity.join(", "),
            types.join(", "),
            result_ty.render(self.dtype, ElemKind::Real),
        ));
        let slope = Value {
            ssa,
            ty: result_ty,
            elem: ElemKind::Real,
        };
        if width == 1 {
            self.reshape(&slope, MlirTy::Ranked(vec![Some(1), d]))
        } else {
            slope
        }
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
        directions: Option<&Directions>,
    ) -> Result<Step, EmitError> {
        let len = step.len();
        let coupled = directions.is_some();
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
                let budget = e.compare("LT", &args[len], limits.max_steps);
                let cap = e.int_value_const(MAX_CONSECUTIVE_REJECTIONS);
                let patience = e.compare("LT", &args[len + 1], &cap);
                let alive = e.not(limits.failed);
                let go = e.and(&open, &budget);
                let go = e.and(&go, &patience);
                e.and(&go, &alive)
            },
            |e, args| {
                let step = Step::from_values(&args[..len], coupled);
                let (t, y, k, h) = (&step.t, &step.y, &step.k, &step.h);
                // The coupled solve's sensitivities and their slope at (t, y).
                let sensitivities = directions.zip(step.sensitivities.clone());
                // A step that would pass the last output time ends on it.
                let end = e.add(t, h);
                let past = e.sub(&end, limits.last);
                let inside = e.compare("LE", &past, limits.eps);
                let to_last = e.sub(limits.last, t);
                let h = e.select(&inside, h, &to_last);
                // Stages k2..k6, then the 5th-order solution and k7 = f(t+h, y5).
                // The sensitivities take the same stages, with slopes from `f`'s
                // Jacobian-vector products at each stage's state.
                let mut ks = vec![k.clone()];
                let mut sks: Vec<Value> = sensitivities
                    .iter()
                    .map(|(_, [_, slope])| slope.clone())
                    .collect();
                let mut s_new = None;
                for (row, (&node, coefficients)) in C.iter().zip(A.iter()).enumerate() {
                    let mut x = y.clone();
                    let mut sx = sensitivities.as_ref().map(|(_, [s, _])| s.clone());
                    for (j, &a) in coefficients.iter().enumerate() {
                        if row == 5 && j == 1 {
                            continue; // b2 = 0 in the solution row.
                        }
                        let a = e.scalar(a);
                        let ha = e.mul(&h, &a);
                        let term = e.mul(&ha, &ks[j]);
                        x = e.add(&x, &term);
                        if let Some(sum) = sx.as_mut() {
                            let term = e.mul(&ha, &sks[j]);
                            *sum = e.add(sum, &term);
                        }
                    }
                    if row == 5 {
                        let t_new = e.add(t, &h);
                        let k7 = e.ode_rhs(id, rhs, &t_new, &x)?;
                        if let (Some((directions, _)), Some(sx)) = (&sensitivities, sx) {
                            sks.push(e.ode_jvp(directions, rhs, &t_new, &x, &sx));
                            s_new = Some(sx);
                        }
                        ks.push(k7);
                        ks.push(x);
                        ks.push(t_new);
                        break;
                    }
                    let c = e.scalar(node);
                    let hc = e.mul(&h, &c);
                    let t_stage = e.add(t, &hc);
                    let kn = e.ode_rhs(id, rhs, &t_stage, &x)?;
                    if let (Some((directions, _)), Some(sx)) = (&sensitivities, &sx) {
                        sks.push(e.ode_jvp(directions, rhs, &t_stage, &x, sx));
                    }
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
                let mut err = e.reduce_max(&ratio);
                // The coupled norm runs over the sensitivities too, each scaled
                // by its own magnitude and slope, as Boost scales any state.
                if let Some((_, [s, s_slope])) = &sensitivities {
                    let mut error: Option<Value> = None;
                    for (&w, kj) in E.iter().zip(&sks) {
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
                    let s_abs = e.abs(s);
                    let slope_abs = e.abs(s_slope);
                    let slope = e.mul(&h_abs, &slope_abs);
                    let scale = e.add(&s_abs, &slope);
                    let scale = e.mul(limits.rtol, &scale);
                    let scale = e.add(limits.atol, &scale);
                    let ratio = e.div(&error, &scale);
                    let s_err = e.reduce_max(&ratio);
                    err = e.max(&err, &s_err);
                }
                let err = e.without_derivative(&err);
                let one = e.scalar(1.0);
                let accept = e.compare("LE", &err, &one);
                // A rejected step has `err > 1`, where `max(err, 1)` is `err`;
                // the clamp keeps the factor finite when it is not selected
                // (a zero error estimate; see the module docs).
                let shrink = {
                    let x = e.max(&err, &one);
                    let p = e.scalar(-1.0 / 3.0);
                    let f = e.pow(&x, &p);
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
                    sensitivities: match (&sensitivities, s_new) {
                        (Some((_, [s, s_slope])), Some(s_new)) => Some([
                            pick(e, &s_new, s),
                            pick(e, sks.last().expect("the end slope"), s_slope),
                        ]),
                        _ => None,
                    },
                };
                let one_int = e.int_value_const(1);
                let zero_int = e.int_value_const(0);
                let accepted = e.add(&args[len], &one_int);
                let accepted = e.select(&accept, &accepted, &args[len]);
                let rejected = e.add(&args[len + 1], &one_int);
                let rejected = e.select(&accept, &zero_int, &rejected);
                let mut values = next.values();
                values.extend([accepted, rejected]);
                Ok(values)
            },
        )?;
        Ok(Step::from_values(&results[..len], coupled))
    }

    /// The step-size controller's view of the error norm `err`: the same value,
    /// with no derivative. StableHLO has no stop-gradient operation (JAX
    /// removes `stop_gradient` before lowering), so the value is rebuilt from
    /// `floor`, whose derivative is zero in every AD system:
    /// `floor(err * 2^64) / 2^64`. The coupled solve applies it elementwise to
    /// the inputs of its Jacobian-vector products and to the combination of
    /// its two solves, where an element below `2^-11` in magnitude moves by
    /// less than `2^-64`.
    ///
    /// The controller cannot tell the result from `err`. For `err >= 2^-12`,
    /// `err * 2^64 >= 2^52` is a whole number, so `floor` keeps it and both
    /// scalings by a power of two are exact. Below `2^-12` the result is in
    /// `[0, err]`, so it stays below the growth floor `5^-5` and both
    /// thresholds, as `err` does. Where `err * 2^64` overflows, the result is
    /// infinite: such a step is rejected with the smallest shrink factor
    /// either way. Zero and NaN are kept.
    fn without_derivative(&mut self, err: &Value) -> Value {
        let scale = self.scalar(2f64.powi(64));
        let scaled = self.mul(err, &scale);
        let whole = self.floor(&scaled);
        self.div(&whole, &scale)
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
    /// A coupled solve's sensitivities `S` and their slope at `(t, y)`, one
    /// row per direction.
    sensitivities: Option<[Value; 2]>,
}

impl Step {
    const LEN: usize = 11;

    fn len(&self) -> usize {
        Self::LEN + if self.sensitivities.is_some() { 2 } else { 0 }
    }

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
        v.extend(self.sensitivities.iter().flatten().cloned());
        v
    }

    fn from_values(v: &[Value], coupled: bool) -> Self {
        Step {
            t: v[0].clone(),
            y: v[1].clone(),
            k: v[2].clone(),
            h: v[3].clone(),
            t_old: v[4].clone(),
            y_old: v[5].clone(),
            k_old: v[6].clone(),
            stages: [v[7].clone(), v[8].clone(), v[9].clone(), v[10].clone()],
            sensitivities: coupled.then(|| [v[11].clone(), v[12].clone()]),
        }
    }
}

/// The inputs of one solve over the output times.
struct Solve<'a> {
    t0: &'a Value,
    y0: &'a Value,
    ts: &'a Value,
    n: u64,
    d: u64,
    rtol: &'a Value,
    atol: &'a Value,
    max_steps: &'a Value,
}

/// The directions of a coupled solve (module docs).
struct Directions {
    /// The number of directions: the non-constant components of `y0` and of
    /// `p`'s real leaves.
    width: u64,
    /// The private function of `f` that `enzyme.fwddiff` differentiates.
    function: String,
    /// `S` at `t0`: `dy0/dy0` in the rows of `y0`'s directions, zero below.
    s0: Value,
    /// Per leaf of `p`: its tangent in every direction, or `None` for a
    /// constant leaf.
    seeds: Vec<Option<Value>>,
}

/// Name prefix of the private right-hand-side functions.
const RHS_FUNCTION: &str = "__flatppl_ode_rhs_";

impl Rhs {
    /// The tensor leaves of the parameter `p`, depth-first.
    fn leaves(&self) -> Vec<Value> {
        self.parameter
            .as_ref()
            .map(|(components, _)| components.values.clone())
            .unwrap_or_default()
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
