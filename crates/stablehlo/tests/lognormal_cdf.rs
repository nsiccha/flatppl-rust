//! LogNormal's CDF lets a truncated LogNormal normalize (§08 support
//! `posreals`). Values against SciPy and Enzyme gradients against central
//! differences were checked on the emitted module (see the commit message).

use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

#[test]
fn truncated_lognormal_normalizes() {
    let src = r#"flatppl_compat = "0.1"
s = elementof(posreals)
mu = elementof(reals)
M = normalize(truncate(LogNormal(mu, 1.0), interval(0.00001, inf)))
M2 = normalize(truncate(LogNormal(mu, 1.0), interval(0.5, 3.0)))
inputs = (s, mu)
outputs = logdensityof(M, s) + logdensityof(M2, s)
"#;
    let mut model = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("determinize");
    for dtype in [Dtype::F32, Dtype::F64] {
        let ir = flatppl_stablehlo::emit(
            &lowered,
            Mode::LogDensity,
            &EmitOptions {
                dtype,
                ..Default::default()
            },
        )
        .expect("a truncated LogNormal has a normalizer");
        assert!(ir.contains("stablehlo.log"), "{ir}");
    }
}
