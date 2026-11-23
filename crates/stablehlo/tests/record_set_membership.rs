//! `in` over a record set with a record-valued point that is not written out:
//! a whole record input, as in `logdensityof(truncate(Lebesgue(S), R), p)` with
//! `p = elementof(S)` (PosteriorDB uk_drivers state space, queried as one
//! record). Each field is tested against the set of the same name.

use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit(src: &str) -> Result<String, String> {
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
    .map_err(|e| e.reason)?;
    flatppl_stablehlo::emit(
        &lowered,
        Mode::LogDensity,
        &EmitOptions {
            dtype: Dtype::F64,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

const TRUNCATED: &str = r#"flatppl_compat = "0.1"
support = cartprod(a = reals, s = ordered(2))
positive = cartprod(a = reals, s = cartpow(posreals, 2))
M = truncate(Lebesgue(support), positive)
"#;

#[test]
fn whole_record_input_is_tested_field_by_field() {
    let whole =
        format!("{TRUNCATED}p = elementof(support)\ninputs = p\noutputs = logdensityof(M, p)\n");
    let written = format!(
        "{TRUNCATED}a = elementof(reals)\ns = elementof(ordered(2))\ninputs = (a, s)\n\
         outputs = logdensityof(M, record(a = a, s = s))\n"
    );
    let whole = emit(&whole).expect("a record input lowers");
    let written = emit(&written).expect("a written record lowers");
    // Both spellings take the same arguments and test positivity of `s`.
    let signature = |ir: &str| {
        ir.lines()
            .find(|l| l.contains("func.func"))
            .unwrap()
            .to_owned()
    };
    assert_eq!(signature(&whole), signature(&written));
    assert!(
        whole.contains("stablehlo.compare  GT") || whole.contains("stablehlo.compare GT"),
        "{whole}"
    );
}
