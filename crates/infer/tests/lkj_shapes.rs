use flatppl_core::{Dim, Mass, ScalarType, Type};

fn inferred(source: &str, name: &str) -> Type {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let (_, binding) = module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)
        .unwrap();
    module.type_of(binding.rhs).unwrap().clone()
}

fn matrix(dimension: Dim) -> Type {
    Type::Array {
        shape: Box::new([dimension, dimension]),
        elem: Box::new(Type::Scalar(ScalarType::Real)),
    }
}

#[test]
fn known_lkj_dimensions_reach_constructors_and_sampling_primitives() {
    for constructor in ["LKJ", "LKJCholesky"] {
        for law in [
            format!("{constructor}(3, 1.5)"),
            format!("{constructor}(eta = 1.5, n = n)"),
        ] {
            let source = format!("start = 2\nn = start + 1\nmeasure = {law}");
            assert_eq!(
                inferred(&source, "measure"),
                Type::Measure {
                    domain: Box::new(matrix(Dim::Static(3))),
                    mass: Mass::Normalized,
                },
                "{law}"
            );
        }
        let source = format!(
            "start = 2\nn = start + 1\nparameters = record(n = n, eta = 1.5)\n\
             result = builtin_sample(rnginit(0), {constructor}, parameters)"
        );
        assert_eq!(
            inferred(&source, "result"),
            Type::Tuple(Box::new([matrix(Dim::Static(3)), Type::RngState]))
        );
    }
}

#[test]
fn unresolved_lkj_dimensions_remain_dynamic() {
    for constructor in ["LKJ", "LKJCholesky"] {
        let source = format!(
            "n = external(posintegers)\nmeasure = {constructor}(n, 1.5)\n\
             result = builtin_sample(rnginit(0), {constructor}, record(n = n, eta = 1.5))"
        );
        assert_eq!(
            inferred(&source, "measure"),
            Type::Measure {
                domain: Box::new(matrix(Dim::Dynamic)),
                mass: Mass::Normalized,
            }
        );
        assert_eq!(
            inferred(&source, "result"),
            Type::Tuple(Box::new([matrix(Dim::Dynamic), Type::RngState]))
        );
    }
}

/// The value set (for a measure, the support) of the binding `name`.
fn valueset(source: &str, name: &str) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let (_, binding) = module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)
        .unwrap();
    let set = module.valueset_of(binding.rhs).unwrap().clone();
    module.display_valueset(&set)
}

/// §08 LKJCholesky's support, "lower-triangular n×n matrices with positive
/// diagonal and unit-norm rows", is the fork's `corrcholesky(n)`: draws, the
/// law and a record prior over it carry the set, sized by `n` as the variate is.
#[test]
fn lkj_cholesky_support_is_corrcholesky() {
    for law in ["LKJCholesky(3, 2.0)", "LKJCholesky(eta = 2.0, n = n)"] {
        let source = format!(
            "start = 2\nn = start + 1\nL ~ {law}\nlaw = lawof(L)\n\
             mu ~ Normal(0.0, 1.0)\nprior = lawof(record(mu = mu, L = L))"
        );
        assert_eq!(valueset(&source, "L"), "corrcholesky(3)", "{law}");
        assert_eq!(valueset(&source, "law"), "corrcholesky(3)", "{law}");
        assert_eq!(
            valueset(&source, "prior"),
            "cartprod(mu = reals, L = corrcholesky(3))",
            "{law}"
        );
    }
    let source = "n = external(posintegers)\nL ~ LKJCholesky(n, 2.0)";
    assert_eq!(valueset(source, "L"), "corrcholesky(?)");
}

/// `elementof(corrcholesky(n))` is an n×n real matrix input, the ABI of
/// `cartpow(reals, [n, n])`; the constraint lives in the value set.
#[test]
fn corrcholesky_elements_are_square_real_matrices() {
    let source = "L = elementof(corrcholesky(4))";
    assert_eq!(inferred(source, "L"), matrix(Dim::Static(4)));
    assert_eq!(valueset(source, "L"), "corrcholesky(4)");
    let source = "S = corrcholesky(2)\nreference = Lebesgue(support = S)";
    assert_eq!(
        inferred(source, "reference"),
        Type::Measure {
            domain: Box::new(matrix(Dim::Static(2))),
            mass: Mass::Finite,
        }
    );
    assert_eq!(valueset(source, "reference"), "corrcholesky(2)");
}
