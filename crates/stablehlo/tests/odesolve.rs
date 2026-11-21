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
    flatppl_stablehlo::emit(
        &lowered,
        Mode::LogDensity,
        &EmitOptions {
            target,
            dtype,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
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
