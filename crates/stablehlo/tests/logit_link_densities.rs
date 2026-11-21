//! A Bernoulli or Binomial probability spelled `invlogit(z)` is scored on the
//! logit scale, `log p = -softplus(-z)` and `log(1 - p) = -softplus(z)`, as
//! Stan's `bernoulli_logit` and `binomial_logit`. Scoring the rounded `p`
//! instead loses `1 - p` entirely from `z` ≈ 17 in f32 and ≈ 37 in f64 (and
//! `p` itself below `z` ≈ -745 in f64), which gave `-inf` with a NaN gradient
//! at finite logits. These tests cannot execute StableHLO, so they check that
//! the returned score no longer depends on the `stablehlo.logistic` result.

use std::collections::{HashMap, HashSet};

use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit(source: &str, dtype: Dtype, restrict: bool, sufficient_statistics: bool) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let options = EmitOptions {
        dtype,
        restrict_enzyme_compatible: restrict,
        sufficient_statistics,
        ..Default::default()
    };
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .unwrap();
    flatppl_stablehlo::emit(&model, Mode::LogDensity, &options).unwrap()
}

fn ssa_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c == '%' || c == '_' || c.is_ascii_alphanumeric()))
        .filter(|token| token.starts_with('%') && token.len() > 1)
}

/// The right-hand side of every op that defines the SSA values the first
/// returned value (the score) depends on.
fn score_cone(hlo: &str) -> Vec<String> {
    let mut defs = HashMap::new();
    for line in hlo.lines() {
        if let Some((lhs, rhs)) = line.trim().split_once(" = ")
            && lhs.starts_with('%')
        {
            defs.insert(lhs.to_owned(), rhs.to_owned());
        }
    }
    let ret = hlo
        .lines()
        .find_map(|line| line.trim().strip_prefix("return "))
        .expect("a return");
    let score = ssa_tokens(ret).next().expect("a returned value").to_owned();
    let mut seen = HashSet::new();
    let mut stack = vec![score];
    let mut cone = Vec::new();
    while let Some(ssa) = stack.pop() {
        if !seen.insert(ssa.clone()) {
            continue;
        }
        if let Some(rhs) = defs.get(&ssa) {
            cone.push(rhs.clone());
            stack.extend(ssa_tokens(rhs).map(str::to_owned));
        }
    }
    cone
}

fn uses(cone: &[String], op: &str) -> bool {
    cone.iter().any(|rhs| rhs.starts_with(&format!("{op} ")))
}

fn every_emission(source: &str, check: impl Fn(&str, &[String])) {
    for dtype in [Dtype::F32, Dtype::F64] {
        for restrict in [true, false] {
            for sufficient_statistics in [true, false] {
                let hlo = emit(source, dtype, restrict, sufficient_statistics);
                check(&hlo, &score_cone(&hlo));
            }
        }
    }
}

fn scores_on_logit_scale(source: &str) {
    every_emission(source, |hlo, cone| {
        assert!(!uses(cone, "stablehlo.logistic"), "{source}\n{hlo}");
        assert!(uses(cone, "stablehlo.log_plus_one"), "{source}\n{hlo}");
    });
}

#[test]
fn bernoulli_of_invlogit_scores_the_logit() {
    for y in ["false", "true"] {
        scores_on_logit_scale(&format!(
            "eta = elementof(reals)\ninputs = eta\n\
             outputs = logdensityof(Bernoulli(invlogit(eta)), {y})\n"
        ));
    }
    // The dotted regression spelling of the posteriordb wells models.
    scores_on_logit_scale(
        "alpha = elementof(reals)\nbeta = elementof(reals)\n\
         x = [0.5, 1.5, 2.5]\n\
         y ~ Bernoulli.(invlogit.(alpha .+ beta .* x))\n\
         K = kernelof(record(y = y), alpha = alpha, beta = beta)\n\
         L = likelihoodof(K, record(y = [true, false, false]))\n\
         inputs = (alpha, beta)\n\
         outputs = logdensityof(L, record(alpha = alpha, beta = beta))\n",
    );
    // The formula importer binds the mean before the response.
    scores_on_logit_scale(
        "a = elementof(reals)\neta = [a, a + 1.0, a - 1.0]\n\
         mu = invlogit.(eta)\ny ~ Bernoulli.(mu)\n\
         K = kernelof(record(y = y), a = a)\n\
         L = likelihoodof(K, record(y = [false, true, false]))\n\
         inputs = a\noutputs = logdensityof(L, record(a = a))\n",
    );
    scores_on_logit_scale(
        "a = elementof(reals)\ninputs = a\n\
         outputs = logdensityof(iid(Bernoulli(invlogit(a)), 4), [false, true, false, false])\n",
    );
    // A selected element of a logistic, with a fixed stand-in probability, is
    // how a grouped likelihood summary scores each group.
    scores_on_logit_scale(
        "a = elementof(reals)\nk = elementof(nonnegintegers)\ninputs = (a, k)\n\
         outputs = logdensityof(Bernoulli(ifelse(k == 0, 0.5, invlogit.([a, a + 1.0])[2])), false)\n",
    );
}

#[test]
fn binomial_of_invlogit_scores_the_logit() {
    scores_on_logit_scale(
        "a = elementof(reals)\ninputs = a\n\
         outputs = logdensityof(Binomial(5, invlogit(a)), 0)\n",
    );
    // A shared probability takes the scalar likelihood summary when enabled.
    scores_on_logit_scale(
        "a = elementof(reals)\ninputs = a\n\
         outputs = logdensityof(iid(Binomial(10, invlogit(a)), 4), [0, 3, 10, 2])\n",
    );
    // The dotted spelling of posteriordb's GLM_Binomial model.
    scores_on_logit_scale(
        "a = elementof(reals)\nb = elementof(reals)\n\
         t = [-1.0, 0.0, 1.0]\nn = [10, 20, 30]\n\
         y ~ Binomial.(n, invlogit.(a .+ b .* t))\n\
         K = kernelof(record(y = y), a = a, b = b)\n\
         L = likelihoodof(K, record(y = [3, 0, 30]))\n\
         inputs = (a, b)\noutputs = logdensityof(L, record(a = a, b = b))\n",
    );
}

/// Controls: a probability that is not literally `invlogit(z)` keeps the
/// probability-scale density, so the checks above are not vacuous.
#[test]
fn other_probabilities_keep_the_probability_scale() {
    for source in [
        "p = elementof(interval(0.0, 1.0))\ninputs = p\n\
         outputs = logdensityof(Bernoulli(p), false)\n",
        "a = elementof(reals)\ninputs = a\n\
         outputs = logdensityof(Bernoulli(0.5 * invlogit(a)), false)\n",
        "a = elementof(reals)\ninputs = a\n\
         outputs = logdensityof(Binomial(5, 0.5 * invlogit(a)), 1)\n",
    ] {
        every_emission(source, |hlo, cone| {
            assert!(
                uses(cone, "stablehlo.log") || uses(cone, "stablehlo.log_plus_one"),
                "{source}\n{hlo}"
            );
            assert!(
                !hlo.contains("stablehlo.logistic") || uses(cone, "stablehlo.logistic"),
                "{source}\n{hlo}"
            );
        });
    }
}
