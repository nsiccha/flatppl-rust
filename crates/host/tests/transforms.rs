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
    // Both use tensor<2x3x...>, but reductions and broadcasts differ by type,
    // so each chart keeps its own: free, constrained and inverse values all
    // have the input's type.
    let options = EmitOptions::default();
    let mut types = Vec::new();
    for query in [&nested, &matrix] {
        let original = serde_json::to_value(query.compile(&options)?)?;
        let input = &original["inputs"][0]["value"];
        let maps = context.transform(query, false)?;
        let free = serde_json::to_value(maps.unconstrained.compile(&options)?)?;
        let forward = serde_json::to_value(maps.constrain.compile(&options)?)?;
        let inverse = serde_json::to_value(maps.unconstrain.compile(&options)?)?;
        assert_eq!(&free["inputs"][0]["value"], input);
        assert_eq!(&forward["output"], input);
        assert_eq!(&inverse["output"], input);
        types.push(input["value_type"].clone());
    }
    assert_ne!(types[0], types[1], "{types:?}");
    Ok(())
}

/// A power of a non-scalar set charts each element and keeps the nesting:
/// the free value of `cartpow(stdsimplex(3), 3)` is three free 2-vectors, of
/// `cartpow(cartpow(reals, 4), 3)` the array itself, inside records, at the
/// top level and in table rows.
#[test]
fn powers_of_array_sets_chart_each_element() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let options = EmitOptions::default();
    let shapes = |schema: &serde_json::Value| {
        schema["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["value"]["shape"].clone())
            .collect::<Vec<_>>()
    };
    let check = |context: &mut Context, source: &str| {
        let query = context.parse(source, None, None)?;
        let original = serde_json::to_value(query.compile(&options)?)?;
        let maps = context.transform(&query, true)?;
        let free = serde_json::to_value(maps.unconstrained.compile(&options)?)?;
        let forward = serde_json::to_value(maps.constrain.compile(&options)?)?;
        let inverse = serde_json::to_value(maps.unconstrain.compile(&options)?)?;
        let input = original["inputs"][0]["value"].clone();
        let coordinates = free["inputs"][0]["value"].clone();
        assert_eq!(forward["inputs"][0]["value"], coordinates);
        assert_eq!(forward["output"], input);
        assert_eq!(inverse["inputs"][0]["value"], input);
        assert_eq!(inverse["output"], coordinates);
        assert_eq!(original["output"], free["output"]);
        Ok::<_, Box<dyn std::error::Error>>((input, coordinates))
    };

    // The HMM and IOHMM shapes: a power of simplices and an array of vectors.
    let (input, free) = check(
        &mut context,
        "p = elementof(cartprod(pi1 = stdsimplex(3), A = cartpow(stdsimplex(3), 3), \
             mu = ordered(3), sigma = cartpow(posreals, 3), w = cartpow(cartpow(reals, 4), 3)))\n\
         inputs = p\n\
         outputs = sum(log.(p.pi1)) + sum(broadcast(r -> sum(log.(r)), p.A)) + sum(p.mu) \
             - sum(p.sigma) - sum(broadcast(r -> sum(r .* r), p.w))",
    )?;
    assert_eq!(
        shapes(&free),
        [json!([2]), json!([3, 2]), json!([3]), json!([3]), json!([3, 4])]
    );
    assert_eq!(
        shapes(&input),
        [json!([3]), json!([3, 3]), json!([3]), json!([3]), json!([3, 4])]
    );
    // The array of vectors is its own chart: free and constrained types agree.
    assert_eq!(free["fields"][4], input["fields"][4]);

    // An LDA-style top-level power of simplices, and of correlation factors.
    for (set, free_shape) in [
        ("cartpow(stdsimplex(2), 6)", json!([6, 1])),
        ("cartpow(stdsimplex(249), 2)", json!([2, 248])),
        ("cartpow(corrcholesky(3), 2)", json!([2, 3])),
        ("cartpow(ordered(3), [2, 2])", json!([2, 2, 3])),
        ("cartpow(cartpow(stdsimplex(3), 2), 2)", json!([2, 2, 2])),
    ] {
        let (_, free) = check(
            &mut context,
            &format!("p = elementof({set})\ninputs = p\noutputs = 0.0"),
        )?;
        assert_eq!(free["shape"], free_shape, "{set}");
    }

    // A table column holding a power of simplices.
    let (_, free) = check(
        &mut context,
        "p = elementof(cartpow(cartprod(A = cartpow(stdsimplex(3), 2), s = posreals), 4))\n\
         inputs = p\noutputs = sum(p.s)",
    )?;
    assert_eq!(free["rows"], 4);
    assert_eq!(shapes(&free), [json!([4, 2, 2]), json!([4])]);
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
