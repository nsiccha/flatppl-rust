//! `odesolve` sensitivity control (`EmitOptions::ode_sensitivity_control`,
//! dogfood-fork extension) through the host's exports: the outputs are lowered
//! again on the coupled steps, as `@logdensity` lowers them. The numbers are
//! checked by executing the exports against BridgeStan (see the commit
//! message); here the structure of the exports is pinned.

use std::path::PathBuf;

use flatppl_host::{BatchSpec, Context, Dtype, EmitOptions, LoadedModule};

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

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "flatppl-host-ode-sensitivity-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn options(ode_sensitivity_control: bool) -> EmitOptions {
    EmitOptions {
        dtype: Dtype::F64,
        ode_sensitivity_control,
        ..Default::default()
    }
}

fn parse(context: &mut Context, name: &str, source: &str) -> std::sync::Arc<LoadedModule> {
    let path = directory(name).join("query.flatppl");
    context.parse(source, None, path.to_str()).unwrap()
}

/// The `enzyme.fwddiff` lines of an export.
fn fwddiffs(stablehlo: &str) -> Vec<&str> {
    stablehlo
        .lines()
        .filter(|line| line.contains("enzyme.fwddiff"))
        .collect()
}

/// The coupled pass's structure: the right-hand side as a private function and
/// one batched product per stage, as in `@logdensity` (`stablehlo` tests).
fn assert_coupled(stablehlo: &str, width: usize) {
    assert!(
        stablehlo.contains("func.func private @__flatppl_ode_rhs_0("),
        "{stablehlo}"
    );
    let products = fwddiffs(stablehlo);
    assert_eq!(products.len(), 7, "{stablehlo}");
    for line in products {
        assert!(line.contains(&format!("width = {width} : i64")), "{line}");
    }
}

#[test]
fn an_export_takes_odesolve_derivatives_on_the_coupled_steps() {
    let mut context = Context::default();
    let module = parse(&mut context, "lv", LOTKA_VOLTERRA);
    let plain = module.compile(&options(false)).unwrap();
    assert!(!plain.stablehlo.contains("enzyme."), "{}", plain.stablehlo);
    let coupled = module.compile(&options(true)).unwrap();
    assert_coupled(&coupled.stablehlo, 6);
    // Each pass nests an adaptive loop in a counted loop over the output times.
    assert_eq!(coupled.stablehlo.matches("stablehlo.while").count(), 4);
    assert_eq!(plain.stablehlo.matches("stablehlo.while").count(), 2);
    // The ABI is the export's own, whichever steps the derivative takes.
    assert_eq!(
        serde_json::to_value(&coupled.inputs).unwrap(),
        serde_json::to_value(&plain.inputs).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&coupled.output).unwrap(),
        serde_json::to_value(&plain.output).unwrap()
    );
}

#[test]
fn a_prepared_export_lowers_its_score_twice() {
    // The regression's summaries move to the preparation; the score keeps the
    // odesolve and is lowered on both step regimes.
    let dir = directory("prepared");
    let rows: Vec<[f64; 2]> = (0..30)
        .map(|i| [1.0, (0.37 * f64::from(i)).sin() * 3.0])
        .collect();
    let y: Vec<f64> = rows.iter().map(|r| 0.5 + 1.5 * r[1]).collect();
    std::fs::write(dir.join("X.json"), serde_json::to_string(&rows).unwrap()).unwrap();
    std::fs::write(dir.join("y.json"), serde_json::to_string(&y).unwrap()).unwrap();
    let source = r#"flatppl_compat = "0.1"
X = load_data("X.json", cartpow(interval(-100.0, 100.0), [30, 2]))
y_data = load_data("y.json", cartpow(interval(-100.0, 100.0), 30))
beta = elementof(cartpow(reals, 2))
sigma = elementof(posreals)
k = elementof(posreals)
decay(t, y, k) = -k .* y
level = sum(rowstack(odesolve(decay, [1.0, 2.0], 0.0, [0.5, 1.0], k))[:, 1])
inputs = (beta, sigma, k)
outputs = logdensityof(Normal.(X * beta, sigma), y_data) + level
"#;
    let path = dir.join("query.flatppl");
    let mut context = Context::default();
    let module = context.parse(source, None, path.to_str()).unwrap();
    let plain = module.compile_prepared(&options(false)).unwrap();
    assert!(plain.preparation.is_some());
    assert!(!plain.stablehlo.contains("enzyme."));
    let coupled = module.compile_prepared(&options(true)).unwrap();
    let preparation = coupled.preparation.as_deref().expect("a preparation");
    assert!(!preparation.stablehlo.contains("enzyme."));
    // One direction, k: the unbatched product.
    let products = fwddiffs(&coupled.stablehlo);
    assert_eq!(products.len(), 7, "{}", coupled.stablehlo);
    assert!(products.iter().all(|line| !line.contains("width =")));
    assert_eq!(
        preparation.stablehlo,
        plain.preparation.as_deref().unwrap().stablehlo
    );
}

#[test]
fn a_batched_export_lowers_its_outputs_twice() {
    let mut context = Context::default();
    let module = parse(&mut context, "batched", LOTKA_VOLTERRA);
    let batch = BatchSpec {
        shape: vec![3],
        input_axes: vec![vec![Some(0)], vec![Some(0)]],
    };
    let plain = module.compile_batched(&options(false), &batch).unwrap();
    assert!(!plain.stablehlo.contains("enzyme."));
    let coupled = module.compile_batched(&options(true), &batch).unwrap();
    assert_eq!(coupled.batch_shape, [3]);
    assert!(
        coupled
            .stablehlo
            .contains("func.func private @__flatppl_ode_rhs_0("),
        "{}",
        coupled.stablehlo
    );
    assert!(!fwddiffs(&coupled.stablehlo).is_empty());
}

#[test]
fn sensitivity_control_leaves_a_module_without_odesolve_unchanged() {
    let mut context = Context::default();
    let module = parse(
        &mut context,
        "plain",
        "x = elementof(reals)\ninputs = x\noutputs = logdensityof(Normal(0.0, 1.0), x)",
    );
    assert_eq!(
        module.compile(&options(true)).unwrap().stablehlo,
        module.compile(&options(false)).unwrap().stablehlo
    );
}

/// Sampling exports (flatppl-python's `autodiff=False`): Enzyme restrictions off.
fn sampling(ode_sensitivity_control: bool) -> EmitOptions {
    EmitOptions {
        restrict_enzyme_compatible: false,
        ..options(ode_sensitivity_control)
    }
}

/// The rng keys an export's draws consume: the operand of each
/// `stablehlo.rng_bit_generator`.
fn consumed_keys(stablehlo: &str) -> Vec<&str> {
    stablehlo
        .lines()
        .filter_map(|line| line.split("stablehlo.rng_bit_generator ").nth(1))
        .map(|rest| rest.split(',').next().unwrap())
        .collect()
}

/// The successor key the `n`th draw (from 0, in emission order) produces.
fn successor(stablehlo: &str, n: usize) -> &str {
    stablehlo
        .lines()
        .filter(|line| line.contains("stablehlo.rng_bit_generator"))
        .nth(n)
        .and_then(|line| line.trim().split(',').next())
        .unwrap()
}

/// The last operand of `@main`'s `return`: the query's final successor key.
fn returned_key(stablehlo: &str) -> &str {
    stablehlo
        .lines()
        .skip_while(|line| !line.contains("func.func @main("))
        .find_map(|line| line.trim().strip_prefix("return "))
        .and_then(|operands| operands.split(" : ").next())
        .and_then(|operands| operands.split(", ").last())
        .unwrap()
}

#[test]
fn a_sampling_export_lowers_its_draw_on_both_step_regimes() {
    // flatppl-python's snag `ode-sensitivity-3b060798`: the successor key
    // reached the combination of the two passes and panicked.
    for law in [
        "m = odesolve(decay, [1.0], 0.0, [1.0], k)\noutputs = rand(Normal(m[1][1], 1.0), state)",
        "outputs = rand(Normal(odesolve(decay, [1.0], 0.0, [1.0], k)[1][1], 1.0), state)",
    ] {
        let source = format!(
            "flatppl_compat = \"0.1\"\ndecay(t, z, th) = [-th * z[1]]\nstate = rnginit(0)\n\
             k = elementof(posreals)\ninputs = (state, k)\n{law}\n"
        );
        let mut context = Context::default();
        let module = parse(&mut context, "draw", &source);
        let plain = module.compile(&sampling(false)).unwrap();
        let coupled = module.compile(&sampling(true)).unwrap();
        // One direction, k: the unbatched product.
        assert!(
            coupled
                .stablehlo
                .contains("func.func private @__flatppl_ode_rhs_0("),
            "{}",
            coupled.stablehlo
        );
        let products = fwddiffs(&coupled.stablehlo);
        assert_eq!(products.len(), 7, "{}", coupled.stablehlo);
        assert!(products.iter().all(|line| !line.contains("width =")));
        // Both passes draw from the input key, and the export returns the
        // value pass's successor.
        assert_eq!(consumed_keys(&plain.stablehlo), ["%arg0"]);
        assert_eq!(consumed_keys(&coupled.stablehlo), ["%arg0", "%arg0"]);
        assert_eq!(
            returned_key(&coupled.stablehlo),
            successor(&coupled.stablehlo, 0),
            "{}",
            coupled.stablehlo
        );
        assert_eq!(
            serde_json::to_value(&coupled.inputs).unwrap(),
            serde_json::to_value(&plain.inputs).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&coupled.output).unwrap(),
            serde_json::to_value(&plain.output).unwrap()
        );
        // The reporter's prepared form, and a batched one.
        module.compile_prepared(&sampling(true)).unwrap();
        let batch = BatchSpec {
            shape: vec![3],
            input_axes: vec![vec![Some(0)], vec![Some(0)]],
        };
        let batched = module.compile_batched(&sampling(true), &batch).unwrap();
        assert!(!fwddiffs(&batched.stablehlo).is_empty());
    }
}

#[test]
fn a_chained_draw_returns_the_value_pass_successor_key() {
    // Both draws are lowered again on the coupled steps; the final successor
    // key, which carries no derivative, is the value pass's.
    let source = "flatppl_compat = \"0.1\"\ndecay(t, z, th) = [-th * z[1]]\n\
                  s = rnginit(0)\nk = elementof(posreals)\n\
                  m = odesolve(decay, [1.0], 0.0, [1.0], k)\n\
                  d, s2 = rand(Normal(m[1][1], 1.0), s)\ny, s3 = rand(Normal(d, 1.0), s2)\n\
                  inputs = (s, k)\noutputs = (y, s3)\n";
    let mut context = Context::default();
    let module = parse(&mut context, "chain", source);
    let plain = module.compile(&sampling(false)).unwrap();
    let coupled = module.compile(&sampling(true)).unwrap();
    assert_eq!(
        consumed_keys(&plain.stablehlo).len(),
        2,
        "{}",
        plain.stablehlo
    );
    assert_eq!(
        consumed_keys(&coupled.stablehlo).len(),
        4,
        "{}",
        coupled.stablehlo
    );
    assert_eq!(
        returned_key(&coupled.stablehlo),
        successor(&coupled.stablehlo, 1),
        "{}",
        coupled.stablehlo
    );
    assert_eq!(
        serde_json::to_value(&coupled.output).unwrap(),
        serde_json::to_value(&plain.output).unwrap()
    );
}
