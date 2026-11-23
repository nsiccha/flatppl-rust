//! Fixed data is held packed: one node per array, whatever its length, and a
//! packed vector lowers exactly as the same `vector` call of literals.

use flatppl_core::{Module, Node, Packed};

fn infer(source: &str, pack: bool) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    if pack {
        assert!(module.pack_literal_vectors() > 0, "nothing to pack");
    }
    let errors = flatppl_infer::infer(&mut module);
    assert!(
        !errors
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error),
        "{errors:?}"
    );
    module
}

fn lower(mut module: Module) -> Module {
    let roots = [module.intern("inputs"), module.intern("outputs")];
    flatppl_determinizer::determinize_with_options(
        &module,
        &Default::default(),
        Some(&roots),
        &flatppl_determinizer::LoweringOptions {
            preserve_rand_tuple: true,
            ..Default::default()
        },
    )
    .unwrap()
}

fn emit(module: &Module) -> String {
    let options = flatppl_stablehlo::EmitOptions {
        dtype: flatppl_stablehlo::Dtype::F64,
        ..Default::default()
    };
    flatppl_stablehlo::emit(module, flatppl_stablehlo::Mode::LogDensity, &options).unwrap()
}

/// SSA values renumbered in order of first use, so programs that differ only in
/// how many dead definitions consumed numbers compare equal.
fn canonical(text: &str) -> String {
    let mut names = std::collections::HashMap::new();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('%') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            out.push('%');
            rest = tail;
            continue;
        }
        let next = names.len();
        let id = *names.entry(&tail[..digits]).or_insert(next);
        out.push_str(&format!("%v{id}"));
        rest = &tail[digits..];
    }
    out.push_str(rest);
    out
}

fn reduced(source: &str, pack: bool) -> String {
    let mut lowered = lower(infer(source, pack));
    flatppl_determinizer::sufficient_statistics(&mut lowered);
    canonical(&emit(&lowered))
}

fn data_source(rows: usize) -> (String, Vec<u8>) {
    let x: Vec<Vec<f64>> = (0..rows)
        .map(|i| (0..3).map(|j| ((i * 3 + j) as f64 * 0.37).sin()).collect())
        .collect();
    let y: Vec<i64> = (0..rows).map(|i| (i % 7) as i64).collect();
    let json = serde_json::json!({ "x": x, "y": y, "n": rows });
    let source = format!(
        "flatppl_compat = \"0.1\"\n\
         d = load_data(\"data.json\", cartprod(x = cartpow(reals, [{rows}, 3]), \
         y = cartpow(nonnegintegers, {rows})))\n\
         beta = elementof(cartpow(reals, 3))\n\
         inputs = beta\n\
         outputs = logdensityof(Poisson.(exp.(d.x * beta)), d.y)\n"
    );
    (source, serde_json::to_vec(&json).unwrap())
}

fn loaded(rows: usize) -> Module {
    let (source, bytes) = data_source(rows);
    let mut lowered = lower(infer(&source, false));
    flatppl_host::load_fixed_data(&mut lowered, |name| {
        assert_eq!(name, "data.json");
        Ok(bytes.clone())
    })
    .unwrap();
    lowered
}

#[test]
fn fixed_data_costs_one_node_per_array_whatever_its_length() {
    let small = loaded(40);
    let large = loaded(1200);
    assert_eq!(small.node_count(), large.node_count());
    let lengths = |m: &Module| {
        let mut lengths = (0..m.node_count())
            .filter_map(|i| {
                match m.node(<flatppl_core::NodeId as flatppl_core::Idx>::from_usize(i)) {
                    Node::Packed(p) if p.len() >= 40 => Some((p.scalar_type(), p.len())),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        lengths.sort_by_key(|&(_, n)| n);
        lengths
    };
    use flatppl_core::ScalarType::{Integer, Real};
    assert_eq!(lengths(&large), [(Integer, 1200), (Real, 3600)]);
    // The data reach the program as one dense constant each.
    let text = emit(&large);
    assert!(text.contains("tensor<3600xf64>"), "{text}");
    assert!(text.contains("tensor<1200x"), "{text}");
}

#[test]
fn packed_vectors_lower_as_their_literal_vectors() {
    // Twelve groups of three fixed observations and two singleton groups.
    let mut g: Vec<usize> = (0..36).map(|i| i / 3 + 1).collect();
    g.extend([13, 14]);
    let y: Vec<String> = (0..g.len())
        .map(|i| format!("{:.4}", (0.7 * i as f64).sin()))
        .collect();
    let g: Vec<String> = g.iter().map(ToString::to_string).collect();
    let partition = format!(
        "mu=elementof(cartpow(reals,14))\nsigma=elementof(posreals)\ng=[{}]\ninputs=(mu,sigma)\noutputs=logdensityof(Normal.(mu[g],sigma),[{}])",
        g.join(","),
        y.join(",")
    );
    let histogram = "lambda=elementof(posreals)\ninputs=lambda\n\
                     outputs=logdensityof(iid(Poisson(lambda),12),[1,2,2,3,1,1,2,0,0,1,2,3])"
        .to_string();
    let repeated = "a=elementof(reals)\nb=elementof(reals)\ninputs=(a,b)\n\
                    x=[0.5,1.5,0.5,1.5,0.5,2.5]\n\
                    outputs=logdensityof(Bernoulli.(invlogit.(a .+ b .* x)),[true,false,true,true,false,true])"
        .to_string();
    let regression =
        "beta=elementof(cartpow(reals,2))\nsigma=elementof(posreals)\ninputs=(beta,sigma)\n\
                      design=rowstack([[1.0,0.3],[1.0,0.7],[1.0,1.9],[1.0,2.4]])\n\
                      outputs=logdensityof(Normal.(design*beta,sigma),[0.1,0.9,2.2,2.6])"
            .to_string();
    let plain = "mu=elementof(reals)\ninputs=mu\n\
                 outputs=logdensityof(Normal.(mu,[1.0,2.0,0.5]),[0.25,-1.5,3.0])"
        .to_string();
    for source in [partition, histogram, repeated, regression, plain] {
        assert_eq!(reduced(&source, false), reduced(&source, true), "{source}");
    }
}

#[test]
fn packed_literals_read_back_as_their_values() {
    let module = infer("x = [1.5, 2.5, 3.0]\ny = [true, false]\n", true);
    for (name, expected) in [
        ("x", Packed::Real([1.5, 2.5, 3.0].into())),
        ("y", Packed::Bool([true, false].into())),
    ] {
        let binding = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == name)
            .unwrap()
            .1;
        assert_eq!(module.node(binding.rhs), &Node::Packed(expected));
    }
    // Printed source spells the array literal the packed node stands for.
    let printed = flatppl_syntax::print(&module);
    assert!(printed.contains("[1.5, 2.5, 3.0]"), "{printed}");
}
