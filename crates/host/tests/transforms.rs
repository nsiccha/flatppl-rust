use flatppl_host::{Context, EmitOptions};
use serde_json::json;

#[test]
fn coordinate_queries_retain_imports_and_change_only_free_dimensions()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse("scale = 2.0", None, None)?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        r#"
        model = load_module("model.flatppl")
        p = elementof(cartprod(cuts = ordered(3), effects = zerosum(3),
                              weights = stdsimplex(4)))
        inputs = p
        outputs = -model.scale * (sum(p.cuts .* p.cuts) + sum(p.effects .* p.effects)
                                  + sum(p.weights .* p.weights))
        "#,
        None,
        None,
    )?;
    let options = EmitOptions::default();
    let original = serde_json::to_value(query.compile(&options)?)?;
    let transformed = context.transform(&query, true)?;
    drop(context);
    let free = serde_json::to_value(transformed.unconstrained.compile(&options)?)?;
    let forward = serde_json::to_value(transformed.constrain.compile(&options)?)?;
    let inverse = serde_json::to_value(transformed.unconstrain.compile(&options)?)?;
    assert_eq!(original["output"], free["output"]);
    let shapes = |schema: &serde_json::Value| {
        schema["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["value"]["shape"].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        shapes(&free["inputs"][0]["value"]),
        [json!([3]), json!([2]), json!([3])]
    );
    assert_eq!(
        shapes(&forward["output"]),
        [json!([3]), json!([3]), json!([4])]
    );
    assert_eq!(forward["output"], inverse["inputs"][0]["value"]);
    assert_eq!(inverse["output"], free["inputs"][0]["value"]);
    assert_eq!(serde_json::to_value(query.compile(&options)?)?, original);
    Ok(())
}

#[test]
fn array_nesting_is_not_silently_replaced_with_matrix_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let nested = context.parse(
        "p = elementof(cartpow(cartpow(reals, 3), 2))\ninputs = p\noutputs = p",
        None,
        None,
    )?;
    let matrix = context.parse(
        "p = elementof(cartpow(reals, [2, 3]))\ninputs = p\noutputs = p",
        None,
        None,
    )?;
    // Both use tensor<2x3x...>, but reductions and broadcasts differ by type.
    nested.compile(&EmitOptions::default())?;
    assert!(context.transform(&nested, false).is_err());
    context
        .transform(&matrix, false)?
        .constrain
        .compile(&EmitOptions::default())?;
    Ok(())
}

#[test]
fn complex_record_fields_add_a_trailing_real_coordinate_pair()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        "p = elementof(cartprod(z = complexes, a = cartpow(complexes, [2, 3])))\n\
         inputs = p\noutputs = -abs2(p.z) - sum(abs2.(p.a))",
        None,
        None,
    )?;
    let maps = context.transform(&query, true)?;
    let options = EmitOptions::default();
    let inverse = serde_json::to_value(maps.unconstrain.compile(&options)?)?;
    let forward = serde_json::to_value(maps.constrain.compile(&options)?)?;
    let density = serde_json::to_value(maps.unconstrained.compile(&options)?)?;
    let coords = &density["inputs"][0]["value"];
    assert_eq!(coords["fields"][0]["value"]["shape"], json!([2]));
    assert_eq!(coords["fields"][1]["value"]["shape"], json!([2, 3, 2]));
    assert_eq!(coords, &inverse["output"]);
    assert_eq!(coords, &forward["inputs"][0]["value"]);
    assert_eq!(forward["output"], inverse["inputs"][0]["value"]);
    Ok(())
}

#[test]
fn table_coordinates_preserve_rows_and_transform_each_field()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        "p = elementof(cartpow(cartprod(gain = complexes, \
             noise = cartprod(scale = posreals, weights = stdsimplex(3))), 4))\n\
         inputs = p\noutputs = -sum(abs2.(p.gain)) - sum(p.noise.scale)",
        None,
        None,
    )?;
    let options = EmitOptions::default();
    let original = serde_json::to_value(query.compile(&options)?)?;
    let maps = context.transform(&query, true)?;
    let forward = serde_json::to_value(maps.constrain.compile(&options)?)?;
    let inverse = serde_json::to_value(maps.unconstrain.compile(&options)?)?;
    let score = serde_json::to_value(maps.unconstrained.compile(&options)?)?;
    let free = &forward["inputs"][0]["value"];
    assert_eq!(free["rows"], 4);
    assert_eq!(free["fields"][0]["value"]["shape"], json!([4, 2]));
    let noise = &free["fields"][1]["value"];
    assert_eq!(noise["rows"], 4);
    assert_eq!(noise["fields"][0]["value"]["shape"], json!([4]));
    assert_eq!(noise["fields"][1]["value"]["shape"], json!([4, 2]));
    assert_eq!(*free, inverse["output"]);
    assert_eq!(*free, score["inputs"][0]["value"]);
    assert_eq!(forward["output"], original["inputs"][0]["value"]);
    assert_eq!(forward["output"], inverse["inputs"][0]["value"]);
    Ok(())
}

/// `corrcholesky(n)` inputs take the LKJCholesky correlation chart: n(n-1)/2
/// free coordinates, an n×n constrained matrix, and an inverse that reads the
/// declared set back. n = 2 and 30 are the PosteriorDB factor sizes.
#[test]
fn corrcholesky_inputs_have_one_free_coordinate_per_strict_lower_entry()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let options = EmitOptions::default();
    for n in [2_u32, 3, 30] {
        let free = n * (n - 1) / 2;
        let query = context.parse(
            &format!(
                "L = elementof(corrcholesky({n}))\ninputs = L\n\
                 outputs = logdensityof(LKJCholesky({n}, 2.0), L)"
            ),
            None,
            None,
        )?;
        let transformed = context.transform(&query, true)?;
        let unconstrained = serde_json::to_value(transformed.unconstrained.compile(&options)?)?;
        let forward = serde_json::to_value(transformed.constrain.compile(&options)?)?;
        let inverse = serde_json::to_value(transformed.unconstrain.compile(&options)?)?;
        let shape = |schema: &serde_json::Value| schema["value"]["shape"].clone();
        assert_eq!(shape(&unconstrained["inputs"][0]), json!([free]), "n = {n}");
        assert_eq!(shape(&forward["inputs"][0]), json!([free]), "n = {n}");
        assert_eq!(forward["output"]["shape"], json!([n, n]), "n = {n}");
        assert_eq!(shape(&inverse["inputs"][0]), json!([n, n]), "n = {n}");
        assert_eq!(inverse["output"]["shape"], json!([free]), "n = {n}");
    }
    // A record field takes the same chart beside its other coordinates.
    let query = context.parse(
        "p = elementof(cartprod(mu = reals, L = corrcholesky(3)))\ninputs = p\n\
         outputs = logdensityof(LKJCholesky(3, 4.0), p.L) - p.mu * p.mu",
        None,
        None,
    )?;
    let transformed = context.transform(&query, true)?;
    let unconstrained = serde_json::to_value(transformed.unconstrained.compile(&options)?)?;
    let fields = unconstrained["inputs"][0]["value"]["fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field["value"]["shape"].clone())
        .collect::<Vec<_>>();
    assert_eq!(fields, [json!([]), json!([3])]);
    // corrcholesky(1) = {[[1]]} has no free coordinate, so no chart.
    let point = context.parse(
        "L = elementof(corrcholesky(1))\ninputs = L\noutputs = sum(L)",
        None,
        None,
    )?;
    assert!(context.transform(&point, false).is_err());
    Ok(())
}
