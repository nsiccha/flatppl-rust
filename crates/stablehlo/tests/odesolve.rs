//! `odesolve` (dogfood-fork extension): inference and StableHLO emission.
//! Numerical accuracy is checked by executing the emitted modules against
//! exact solutions and BridgeStan (see the commit message); here the shape of
//! the lowering and the refusals are pinned.

use flatppl_core::{Dim, ScalarType, Type};
use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

const LOTKA_VOLTERRA: &str = r#"flatppl_compat = "0.1"
ts = [1.0, 2.0, 3.0, 4.0]
theta = elementof(cartpow(posreals, 4))
z0 = elementof(cartpow(posreals, 2))
predator_prey(t, z, th) = [(th[1] - th[2] * z[2]) * z[1], (-th[3] + th[4] * z[1]) * z[2]]
solution = odesolve(predator_prey, z0, 0.0, ts, theta, rtol = 1e-5, atol = 1e-3, max_steps = 500)
traj = rowstack(solution)
inputs = (theta, z0)
outputs = sum(traj[:, 1]) + sum(traj[:, 2])
"#;

/// The one-compartment Michaelis–Menten absorption model's call: a record
/// parameter, a time-dependent branch and no explicit `max_steps`.
const ABSORPTION: &str = r#"flatppl_compat = "0.1"
times = [0.5, 1.0, 1.5, 2.0]
k_a = elementof(posreals)
V_m = elementof(posreals)
absorption(t, z, p) = [(ifelse(t > 0, exp(-p.k_a * t) * p.D * p.k_a, 0.0) -
    p.V_m * z[1] / (p.K_m + z[1])) / p.V]
C = rowstack(odesolve(absorption, [0.0], 0.0, times,
    record(k_a = k_a, K_m = 1.0, V_m = V_m, D = 30.0, V = 2.0), rtol = 1e-10, atol = 1e-10))
inputs = (k_a, V_m)
outputs = sum(C[:, 1])
"#;

const DECAY: &str = r#"flatppl_compat = "0.1"
k = elementof(posreals)
decay(t, y) = -0.7 .* y
outputs = sum(rowstack(odesolve(decay, [k, 2.0], 0.0, [0.5, 1.0]))[:, 1])
inputs = k
"#;

/// The soil incubation model's call: `y0` is built from record fields, whose
/// element type is still undecided when the rule first runs.
const TWO_POOL: &str = r#"flatppl_compat = "0.1"
domain = cartprod(k1 = posreals, k2 = posreals, alpha21 = posreals, alpha12 = posreals,
    gamma = unitinterval)
p = elementof(domain)
two_pool(t, C, th) = [-th.k1 * C[1] + th.alpha12 * th.k2 * C[2],
    -th.k2 * C[2] + th.alpha21 * th.k1 * C[1]]
pools = rowstack(odesolve(two_pool, [p.gamma * 7.7, (1 - p.gamma) * 7.7], 0.0,
    [1.0, 2.0, 5.0], p, rtol = 1e-10, atol = 1e-10))
inputs = p
outputs = sum(pools[:, 1] .+ pools[:, 2])
"#;

fn inferred(src: &str) -> (flatppl_core::Module, Vec<flatppl_infer::Diagnostic>) {
    let mut model = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    (model, diagnostics)
}

fn emit(src: &str, target: Target, dtype: Dtype) -> Result<String, String> {
    emit_with(
        src,
        &EmitOptions {
            target,
            dtype,
            ..Default::default()
        },
    )
}

/// `f64` CPU emission with odesolve's sensitivity control.
fn emit_coupled(src: &str) -> Result<String, String> {
    emit_with(
        src,
        &EmitOptions {
            dtype: Dtype::F64,
            ode_sensitivity_control: true,
            ..Default::default()
        },
    )
}

fn emit_with(src: &str, options: &EmitOptions) -> Result<String, String> {
    let (mut model, diagnostics) = inferred(src);
    if let Some(d) = diagnostics.first() {
        return Err(d.message.clone());
    }
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .map_err(|e| e.reason)?;
    flatppl_stablehlo::emit(&lowered, Mode::LogDensity, options).map_err(|e| e.to_string())
}

#[test]
fn odesolve_result_is_one_state_per_output_time() {
    let (model, diagnostics) = inferred(LOTKA_VOLTERRA);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let (_, binding) = model
        .bindings()
        .find(|(_, b)| model.resolve(b.name) == "solution")
        .unwrap();
    assert_eq!(
        model.type_of(binding.rhs).unwrap(),
        &Type::Array {
            shape: Box::new([Dim::Static(4)]),
            elem: Box::new(Type::Array {
                shape: Box::new([Dim::Static(2)]),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            }),
        }
    );
}

#[test]
fn mixed_positional_and_keyword_call_lowers_to_nested_loops() {
    for target in [Target::Cpu, Target::Gpu] {
        for dtype in [Dtype::F32, Dtype::F64] {
            let ir = emit(LOTKA_VOLTERRA, target, dtype).expect("emit odesolve");
            // A counted loop over the output times around one adaptive loop.
            assert_eq!(ir.matches("stablehlo.while").count(), 2, "{ir}");
            assert!(ir.contains("tensor<4x2x"), "{ir}");
        }
    }
}

#[test]
fn record_parameter_branch_and_two_input_callback_lower() {
    for src in [ABSORPTION, DECAY, TWO_POOL] {
        let ir = emit(src, Target::Cpu, Dtype::F64).expect("emit odesolve");
        assert_eq!(ir.matches("stablehlo.while").count(), 2, "{ir}");
    }
}

/// The operands of the IR line that defines `name`.
fn definition<'a>(ir: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    ir.lines()
        .map(str::trim)
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no definition of {name}"))
}

/// The IR lines other than its definition that read `name`.
fn uses<'a>(ir: &'a str, name: &str) -> Vec<&'a str> {
    let prefix = format!("{name} = ");
    ir.lines()
        .map(str::trim)
        .filter(|line| !line.starts_with(&prefix))
        .filter(|line| {
            line.split(|c: char| !(c == '%' || c == '_' || c.is_ascii_alphanumeric()))
                .any(|token| token == name)
        })
        .collect()
}

fn defined_name(line: &str) -> &str {
    line.split(" = ").next().unwrap()
}

/// The step-size controller reads the error norm only as
/// `floor(err * 2^64) / 2^64`: the same value without a derivative, so step
/// sizes are not differentiated.
#[test]
fn step_size_controller_reads_the_error_norm_without_its_derivative() {
    for dtype in [Dtype::F32, Dtype::F64] {
        let ir = emit(LOTKA_VOLTERRA, Target::Cpu, dtype).expect("emit odesolve");
        let norms = ir
            .lines()
            .map(str::trim)
            .filter(|line| line.contains("applies stablehlo.maximum"))
            .collect::<Vec<_>>();
        assert_eq!(norms.len(), 1, "{ir}");
        let norm = defined_name(norms[0]);
        let [scaled] = uses(&ir, norm)[..] else {
            panic!("the error norm {norm} must have one use: {ir}");
        };
        assert!(scaled.contains("stablehlo.multiply"), "{scaled}");
        let scale = scaled
            .split([',', ' '])
            .find(|token| {
                token.starts_with('%') && *token != norm && *token != defined_name(scaled)
            })
            .unwrap();
        assert!(
            definition(&ir, scale).contains("dense<18446744073709552000.0>"),
            "{}",
            definition(&ir, scale)
        );
        let [whole] = uses(&ir, defined_name(scaled))[..] else {
            panic!("{ir}");
        };
        assert!(whole.contains("stablehlo.floor"), "{whole}");
        let [frozen] = uses(&ir, defined_name(whole))[..] else {
            panic!("{ir}");
        };
        assert!(
            frozen.contains("stablehlo.divide") && frozen.contains(scale),
            "{frozen}"
        );
        // The controller (accept test, shrink and growth factors) reads it.
        assert!(uses(&ir, defined_name(frozen)).len() >= 3, "{ir}");
    }
}

/// The operation that defines SSA value `name` in `ir`.
fn defining_op<'a>(ir: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    ir.lines()
        .map(str::trim_start)
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .and_then(|rhs| rhs.split([' ', '(']).next())
        .unwrap_or_else(|| panic!("no definition of {name}\n{ir}"))
}

/// Both step-size factors are computed on every step and one is selected, so
/// each `power` must read a clamped error estimate: an unclamped
/// `err^(-1/3)` is infinite at a zero estimate (a steady state), and reverse
/// mode then multiplies the unselected branch's zero cotangent by an infinite
/// slope, giving NaN gradients.
#[test]
fn step_size_factors_read_a_clamped_error_estimate() {
    for dtype in [Dtype::F32, Dtype::F64] {
        // These callbacks contain no `pow`, so every power is the controller's.
        for src in [LOTKA_VOLTERRA, DECAY, TWO_POOL] {
            let ir = emit(src, Target::Cpu, dtype).expect("emit odesolve");
            let powers: Vec<&str> = ir
                .lines()
                .filter_map(|line| line.split_once("= stablehlo.power ").map(|(_, rhs)| rhs))
                .collect();
            assert_eq!(powers.len(), 2, "{ir}");
            for rhs in powers {
                let base = rhs.split(',').next().unwrap().trim();
                assert_eq!(
                    defining_op(&ir, base),
                    "stablehlo.maximum",
                    "power base {base} is not clamped\n{ir}"
                );
            }
        }
    }
}

#[test]
fn unknown_keyword_and_non_vector_state_are_refused() {
    let unknown = LOTKA_VOLTERRA.replace("max_steps = 500", "method = 1");
    let error = emit(&unknown, Target::Cpu, Dtype::F64).unwrap_err();
    assert!(error.contains("no keyword input `method`"), "{error}");

    let scalar = DECAY.replace("[k, 2.0]", "k");
    let error = emit(&scalar, Target::Cpu, Dtype::F64).unwrap_err();
    assert!(
        error.contains("initial state `y0` as a real vector"),
        "{error}"
    );

    let real_steps = LOTKA_VOLTERRA.replace("max_steps = 500", "max_steps = 0.5");
    let error = emit(&real_steps, Target::Cpu, Dtype::F64).unwrap_err();
    assert!(error.contains("`max_steps` as an integer"), "{error}");
}

/// The `enzyme.fwddiff` lines of an emitted module.
fn fwddiffs(ir: &str) -> Vec<&str> {
    ir.lines()
        .map(str::trim)
        .filter(|line| line.contains("enzyme.fwddiff"))
        .collect()
}

#[test]
fn sensitivity_control_is_off_by_default() {
    let ir = emit(LOTKA_VOLTERRA, Target::Cpu, Dtype::F64).expect("emit odesolve");
    assert!(!ir.contains("enzyme."), "{ir}");
    assert!(!ir.contains("__flatppl_ode_rhs_"), "{ir}");
}

#[test]
fn sensitivity_control_lowers_the_outputs_again_on_coupled_steps() {
    let ir = emit_coupled(LOTKA_VOLTERRA).expect("emit odesolve");
    // The value pass and the coupled pass each nest an adaptive loop in a
    // counted loop over the output times.
    assert_eq!(ir.matches("stablehlo.while").count(), 4, "{ir}");
    assert!(
        ir.contains(
            "func.func private @__flatppl_ode_rhs_0(%t: tensor<f64>, %y: tensor<2xf64>, \
             %p0: tensor<4xf64>) -> tensor<2xf64>"
        ),
        "{ir}"
    );
    // z0's 2 and theta's 4 components are the directions: one product at
    // t0 and one per stage after the first.
    let products = fwddiffs(&ir);
    assert_eq!(products.len(), 7, "{ir}");
    for line in products {
        assert!(line.contains("@__flatppl_ode_rhs_0("), "{line}");
        assert!(line.contains("width = 6 : i64"), "{line}");
        assert!(
            line.contains(
                "activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>, \
                 #enzyme.activity<enzyme_dup>]"
            ),
            "{line}"
        );
        assert!(
            line.ends_with(
                "(tensor<f64>, tensor<2xf64>, tensor<6x2xf64>, tensor<4xf64>, \
                 tensor<6x4xf64>) -> tensor<6x2xf64>"
            ),
            "{line}"
        );
    }
    // The coupled loops carry the sensitivities and their slope.
    assert!(
        ir.contains("tensor<6x2xf64>, tensor<6x2xf64>, tensor<4x2xf64>"),
        "{ir}"
    );
}

#[test]
fn sensitivity_directions_skip_constant_parameters_and_states() {
    // `[0.0]` and the record's K_m, D and V are constants: k_a and V_m remain.
    let ir = emit_coupled(ABSORPTION).expect("emit odesolve");
    let products = fwddiffs(&ir);
    assert_eq!(products.len(), 7, "{ir}");
    let const_ = "#enzyme.activity<enzyme_const>";
    let dup = "#enzyme.activity<enzyme_dup>";
    let activity =
        format!("activity=[{const_}, {dup}, {dup}, {const_}, {dup}, {const_}, {const_}]");
    for line in products {
        assert!(line.contains(&activity), "{line}");
        assert!(line.contains("width = 2 : i64"), "{line}");
    }
}

#[test]
fn one_sensitivity_direction_takes_the_unbatched_form() {
    let src = r#"flatppl_compat = "0.1"
k = elementof(posreals)
decay(t, y, k) = -k .* y
outputs = sum(rowstack(odesolve(decay, [1.0, 2.0], 0.0, [0.5, 1.0], k))[:, 1])
inputs = k
"#;
    let ir = emit_coupled(src).expect("emit odesolve");
    let products = fwddiffs(&ir);
    assert_eq!(products.len(), 7, "{ir}");
    for line in products {
        assert!(!line.contains("width ="), "{line}");
        assert!(
            line.ends_with("(tensor<f64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<2xf64>"),
            "{line}"
        );
    }
}

#[test]
fn sensitivity_control_without_directions_changes_nothing() {
    let src = r#"flatppl_compat = "0.1"
x = elementof(reals)
decay(t, y, k) = -k .* y
outputs = x * sum(rowstack(odesolve(decay, [1.0, 2.0], 0.0, [0.5, 1.0], 0.7))[:, 1])
inputs = x
"#;
    let coupled = emit_coupled(src).expect("emit odesolve");
    let plain = emit(src, Target::Cpu, Dtype::F64).expect("emit odesolve");
    assert_eq!(coupled, plain);
}

#[test]
fn sample_mode_refuses_sensitivity_control() {
    // A draw whose mean is read from the solution.
    let src = r#"flatppl_compat = "0.1"
k = elementof(posreals)
decay(t, y, k) = -k .* y
level = sum(rowstack(odesolve(decay, [1.0, 2.0], 0.0, [0.5, 1.0], k))[:, 1])
draw, state = rand(Normal(level, 1.0), rnginit(7))
inputs = k
outputs = (draw, state)
"#;
    let (mut model, diagnostics) = inferred(src);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    let options = EmitOptions {
        dtype: Dtype::F64,
        restrict_enzyme_compatible: false,
        ode_sensitivity_control: true,
        ..Default::default()
    };
    let error = flatppl_stablehlo::emit(&lowered, Mode::Sample, &options)
        .expect_err("sample mode refuses sensitivity control")
        .to_string();
    assert!(error.contains("not to sample mode"), "{error}");
    let plain = EmitOptions {
        ode_sensitivity_control: false,
        ..options
    };
    flatppl_stablehlo::emit(&lowered, Mode::Sample, &plain).expect("sample mode without it");
}
