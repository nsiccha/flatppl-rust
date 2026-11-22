//! A prepared compile runs the data-only setup of fixed-data likelihood
//! summaries once, in `Export::preparation`, instead of on every call.

use std::path::PathBuf;

use flatppl_host::{BatchSpec, Context, Dtype, EmitOptions, Export, LoadedModule, Schema};

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "flatppl-host-prepared-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn options() -> EmitOptions {
    EmitOptions {
        dtype: Dtype::F64,
        ..Default::default()
    }
}

fn leaves(schema: &Schema) -> usize {
    match schema {
        Schema::Tensor { .. } => 1,
        Schema::Record { fields } | Schema::Table { fields, .. } => {
            fields.iter().map(|f| leaves(&f.value)).sum()
        }
        Schema::Tuple { items } => items.iter().map(leaves).sum(),
    }
}

/// Arguments of the entry point.
fn arguments(stablehlo: &str) -> usize {
    let main = stablehlo
        .lines()
        .find(|line| line.contains("func.func @main"))
        .unwrap();
    main.split("->").next().unwrap().matches("%arg").count()
}

fn codes(export: &Export) -> Vec<i64> {
    export
        .runtime_status
        .as_ref()
        .map_or_else(Vec::new, |status| status.codes.clone())
}

/// A Gaussian regression on 40 fixed rows: an intercept and two features.
fn regression(context: &mut Context, name: &str) -> std::sync::Arc<LoadedModule> {
    let dir = directory(name);
    let rows: Vec<[f64; 3]> = (0..40)
        .map(|i| {
            let t = f64::from(i);
            [
                1.0,
                (0.37 * t).sin() * 3.0,
                (0.11 * t).cos() * 2.0 + 0.1 * t,
            ]
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| 0.5 + 1.5 * r[1] - 0.7 * r[2] + 0.3 * (1.3 * i as f64).sin())
        .collect();
    std::fs::write(dir.join("X.json"), serde_json::to_string(&rows).unwrap()).unwrap();
    std::fs::write(dir.join("y.json"), serde_json::to_string(&y).unwrap()).unwrap();
    let source = "n = 40\np = 3\n\
        X = load_data(\"X.json\", cartpow(interval(-100.0, 100.0), [n, p]))\n\
        y_data = load_data(\"y.json\", cartpow(interval(-100.0, 100.0), n))\n\
        beta = elementof(cartpow(reals, p))\nsigma = elementof(posreals)\n\
        inputs = (beta, sigma)\n\
        outputs = logdensityof(Normal.(X * beta, sigma), y_data)";
    let path = dir.join("query.flatppl");
    context.parse(source, None, path.to_str()).unwrap()
}

#[test]
fn fixed_regression_setup_runs_in_the_preparation_only() {
    let mut context = Context::default();
    let module = regression(&mut context, "regression");
    let ordinary = module.compile(&options()).unwrap();
    assert!(ordinary.preparation.is_none());
    // Without preparation, every call repeats the bounded QR setup.
    assert!(ordinary.stablehlo.contains("stablehlo.while"));

    let prepared = module.compile_prepared(&options()).unwrap();
    let preparation = prepared.preparation.as_deref().expect("a preparation");
    assert!(preparation.inputs.is_empty());
    assert!(preparation.stablehlo.contains("stablehlo.while"));
    assert!(!prepared.stablehlo.contains("stablehlo.while"));
    // The declared ABI is unchanged; prepared leaves follow it.
    let names: Vec<_> = prepared.inputs.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["beta", "sigma"]);
    let declared: usize = prepared.inputs.iter().map(|f| leaves(&f.value)).sum();
    assert_eq!(declared, 2);
    let summaries = leaves(&preparation.output);
    assert!(summaries > 0);
    assert_eq!(arguments(&prepared.stablehlo), declared + summaries);
    assert_eq!(arguments(&ordinary.stablehlo), declared);
    // Data-only checks report once, during preparation.
    for code in codes(preparation) {
        assert!(
            codes(&ordinary).contains(&code),
            "preparation code {code} is new"
        );
    }
    let json = serde_json::to_value(&prepared).unwrap();
    assert!(json["preparation"]["stablehlo"].is_string());
    assert!(
        serde_json::to_value(&ordinary)
            .unwrap()
            .get("preparation")
            .is_none()
    );
}

#[test]
fn prepared_leaves_are_shared_by_batch_cells() {
    let mut context = Context::default();
    let module = regression(&mut context, "batched");
    let prepared = module.compile_prepared(&options()).unwrap();
    let summaries = leaves(&prepared.preparation.as_deref().unwrap().output);
    let mut input_axes = vec![vec![Some(0)], vec![Some(0)]];
    input_axes.extend(std::iter::repeat_n(vec![None], summaries));
    let batch = BatchSpec {
        shape: vec![3],
        input_axes: input_axes.clone(),
    };
    let batched = module.compile_batched_prepared(&options(), &batch).unwrap();
    assert!(batched.preparation.is_some());
    assert!(!batched.stablehlo.contains("stablehlo.while"));
    assert_eq!(batched.batch_shape, [3]);
    input_axes[2] = vec![Some(0)];
    let error = module
        .compile_batched_prepared(
            &options(),
            &BatchSpec {
                shape: vec![3],
                input_axes,
            },
        )
        .err()
        .expect("a mapped prepared leaf is refused");
    assert!(
        error.message.contains("take no batch axis"),
        "{}",
        error.message
    );
}

#[test]
fn a_query_without_summaries_compiles_as_before() {
    let mut context = Context::default();
    let module = context
        .parse(
            "x = elementof(reals)\ninputs = x\noutputs = logdensityof(Normal(0.0, 1.0), x)",
            None,
            None,
        )
        .unwrap();
    let ordinary = module.compile(&options()).unwrap();
    let prepared = module.compile_prepared(&options()).unwrap();
    assert!(prepared.preparation.is_none());
    assert_eq!(prepared.stablehlo, ordinary.stablehlo);
    let disabled = EmitOptions {
        sufficient_statistics: false,
        ..options()
    };
    let module = regression(&mut context, "disabled");
    let prepared = module.compile_prepared(&disabled).unwrap();
    assert!(prepared.preparation.is_none());
    assert_eq!(
        prepared.stablehlo,
        module.compile(&disabled).unwrap().stablehlo
    );
}
