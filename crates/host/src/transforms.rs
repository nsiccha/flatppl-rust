//! Canonical coordinates for continuous domains, expressed with existing FlatPPL operations.

use std::sync::Arc;

use flatppl_core::{Module, Type};

use crate::{Domain, DomainField, LoadedModule};

/// Three query modules accompanying an unchanged constrained query.
pub struct TransformedQueries {
    /// The original output evaluated in free coordinates, optionally with a density correction.
    pub unconstrained: Arc<LoadedModule>,
    /// Map free coordinates to the declared domain.
    pub constrain: Arc<LoadedModule>,
    /// Map interior domain values to free coordinates, without clipping.
    pub unconstrain: Arc<LoadedModule>,
}

pub(crate) struct Chart {
    pub domain: String,
    pub free: String,
    pub constrain: String,
    pub unconstrain: String,
    pub log_det: String,
    size: u32,
    free_size: u32,
    scalar: bool,
}

fn vector_range(point: &str, start: u32, end: u32) -> String {
    vector_select(point, start..=end)
}

fn vector_select(point: &str, indices: impl Iterator<Item = u32>) -> String {
    let indices = indices
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{point}[[{indices}]]")
}

fn sum(parts: impl Iterator<Item = String>) -> String {
    let parts: Vec<_> = parts.filter(|part| part != "0.0").collect();
    if parts.is_empty() {
        "0.0".into()
    } else {
        parts.join(" + ")
    }
}

/// A fresh name for the cell variable of a broadcast over `point`: one more
/// than the deepest `row<k>` the point already names, so a nested broadcast
/// never captures an enclosing cell.
fn fresh_row(point: &str) -> String {
    let deepest = point
        .match_indices("row")
        .filter_map(|(i, _)| {
            let digits: String = point[i + 3..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse::<u32>().ok()
        })
        .max()
        .unwrap_or(0);
    format!("row{}", deepest + 1)
}

// Inspection metadata flattens nested Cartesian powers, so charts read the
// nesting back from the value's type (`nest`) rather than rebuilding those as
// matrices: their broadcast semantics differ even when the ABI agrees.
pub(crate) fn supported_container(ty: &Type) -> bool {
    match ty {
        Type::Scalar(_) => true,
        Type::Array { elem, .. } => match elem.as_ref() {
            Type::Scalar(_) => true,
            inner @ Type::Array { .. } => supported_container(inner),
            _ => false,
        },
        Type::Record(fields)
        | Type::Table {
            columns: fields, ..
        } => fields.iter().all(|(_, ty)| supported_container(ty)),
        _ => false,
    }
}

/// `domain` with the nesting of `ty` restored. [`Domain::of`] writes
/// `cartpow(cartpow(reals, 4), 3)` and `cartpow(reals, [3, 4])` as one
/// flattened power; the type tells the array of vectors from the matrix.
/// Records and table rows are restored field by field.
pub(crate) fn nest(domain: &Domain, ty: &Type, module: &Module) -> Result<Domain, String> {
    match (ty, domain) {
        (Type::Array { shape, elem }, Domain::CartPow { of, shape: flat })
            if !matches!(elem.as_ref(), Type::Scalar(_)) =>
        {
            let outer = shape.len();
            let inner = match flat.len() {
                n if n == outer => of.as_ref().clone(),
                n if n > outer => Domain::CartPow {
                    of: of.clone(),
                    shape: flat[outer..].to_vec(),
                },
                _ => return Err("the input's domain has fewer axes than its type".into()),
            };
            Ok(Domain::CartPow {
                of: Box::new(nest(&inner, elem, module)?),
                shape: flat[..outer].to_vec(),
            })
        }
        (Type::Table { columns, .. }, Domain::CartPow { of, shape }) => Ok(Domain::CartPow {
            of: Box::new(nest(of, &Type::Record(columns.clone()), module)?),
            shape: shape.clone(),
        }),
        (Type::Record(types), _) => {
            let Some(fields) = record_fields(domain) else {
                return Ok(domain.clone());
            };
            let fields = fields
                .into_iter()
                .map(|field| {
                    let ty = types
                        .iter()
                        .find(|(name, _)| module.resolve(*name) == field.name);
                    Ok(DomainField {
                        set: match ty {
                            Some((_, ty)) => nest(&field.set, ty, module)?,
                            None => field.set,
                        },
                        name: field.name,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Domain::Record { fields })
        }
        _ => Ok(domain.clone()),
    }
}

fn record_fields(domain: &Domain) -> Option<Vec<DomainField>> {
    match domain {
        Domain::Record { fields } => Some(fields.clone()),
        Domain::CartProd { of } => Some(
            of.iter()
                .map(record_fields)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect(),
        ),
        _ => None,
    }
}

fn record_chart(
    fields: &[DomainField],
    point: &str,
    correlations: &[String],
) -> Result<Chart, String> {
    for name in correlations {
        if !fields.iter().any(|field| &field.name == name) {
            return Err(format!(
                "no correlation factor named `{name}` in the input record"
            ));
        }
    }
    let fields = fields
        .iter()
        .map(|field| {
            let access = format!("{point}.{}", field.name);
            let chart = if correlations.contains(&field.name) {
                correlation_chart(&field.set, &access)?
            } else {
                field.set.chart(&access)?
            };
            Ok((&field.name, chart))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let join = |value: fn(&Chart) -> &String| {
        fields
            .iter()
            .map(|(name, chart)| format!("{name} = {}", value(chart)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Ok(Chart {
        domain: format!("cartprod({})", join(|c| &c.domain)),
        free: format!("cartprod({})", join(|c| &c.free)),
        constrain: format!("record({})", join(|c| &c.constrain)),
        unconstrain: format!("record({})", join(|c| &c.unconstrain)),
        log_det: sum(fields.iter().map(|(_, c)| c.log_det.clone())),
        // Record products concatenate fields, never tensor slices.
        size: 0,
        free_size: 0,
        scalar: false,
    })
}

// Formula correlation factors use LKJCholesky, whose reference measure is
// Lebesgue measure on the independent strictly lower-triangular coordinates.
// Such a factor is declared as a square real matrix; `corrcholesky(n)` inputs
// take the same chart through `Domain::chart`.
fn correlation_chart(domain: &Domain, point: &str) -> Result<Chart, String> {
    if let Domain::CorrCholesky { .. } = domain {
        return domain.chart(point);
    }
    let Domain::CartPow { of, shape } = domain else {
        return Err(format!(
            "correlation factor `{point}` must be a square real matrix"
        ));
    };
    let [Some(n), Some(m)] = shape.as_slice() else {
        return Err(format!(
            "correlation factor `{point}` must have a known matrix shape"
        ));
    };
    let n = *n;
    if n != *m || n < 2 || !matches!(of.as_ref(), Domain::Reals) {
        return Err(format!(
            "correlation factor `{point}` must be a square real matrix of size at least two"
        ));
    }
    cholesky_factor_chart(n, format!("cartpow(reals, [{n}, {n}])"), point)
}

/// The chart of n×n correlation Cholesky factors (n >= 2) declared on
/// `domain`: tanh partial correlations, with free coordinates in the row-major
/// order of the n(n-1)/2 strictly lower-triangular entries. Its log volume is
/// relative to coordinate Lebesgue measure on those entries, LKJCholesky's
/// reference measure (§08) and `Lebesgue(support = corrcholesky(n))`'s.
fn cholesky_factor_chart(n: u32, domain: String, point: &str) -> Result<Chart, String> {
    let size = n
        .checked_mul(n)
        .ok_or("correlation matrix size overflows")?;
    let free_size = (size - n) / 2;

    let mut rows = vec![format!("cat([1.0], zeros({}))", n - 1)];
    let mut inverse = Vec::new();
    let mut determinants = Vec::new();
    let mut offset = 1;
    for row in 2..=n {
        let u = vector_range(point, offset, offset + row - 2);
        offset += row - 1;
        // tanh(u) gives a partial correlation and exp(log_sech) the
        // remaining row norm. Compute log_sech without squaring rounded tanh.
        let magnitude = format!("abs.({u})");
        let log_sech = format!(
            "({} .- {magnitude} .- log1p.(exp.(-2.0 .* {magnitude})))",
            std::f64::consts::LN_2
        );
        let prefix = format!("(cumsum({log_sech}))");
        let before = if row == 2 {
            "[1.0]".into()
        } else {
            format!("cat([1.0], exp.({}))", vector_range(&prefix, 1, row - 2))
        };
        let zeros = if row < n {
            format!(", zeros({})", n - row)
        } else {
            String::new()
        };
        rows.push(format!(
            "cat(tanh.({u}) .* {before}, [exp({prefix}[{}])]{zeros})",
            row - 1
        ));

        // Each coordinate contributes its tanh derivative (sech^2), plus
        // one sech factor to every later strict-lower entry in its row.
        let powers = (2..=row).rev().map(|p| p.to_string()).collect::<Vec<_>>();
        determinants.push(format!("sum([{}] .* {log_sech})", powers.join(", ")));

        for col in 1..row {
            // asinh(entry / remaining_norm) avoids subtracting a prefix of
            // squares from one. Scale the tail before taking its norm.
            let tail = (col + 1..=row)
                .map(|i| format!("{point}[{row}, {i}]"))
                .collect::<Vec<_>>();
            let remaining = if col + 1 == row {
                tail[0].clone()
            } else {
                let tail = format!("[{}]", tail.join(", "));
                let scale = format!("linfnorm({tail})");
                format!("({scale} * l2norm({tail} ./ {scale}))")
            };
            inverse.push(format!("asinh({point}[{row}, {col}] / {remaining})"));
        }
    }
    Ok(Chart {
        domain,
        free: format!("cartpow(reals, {free_size})"),
        constrain: format!("rowstack([{}])", rows.join(", ")),
        unconstrain: format!(
            "ifelse(isfinite(logdensityof(LKJCholesky({n}, 1.0), {point})), [{}], fill(-inf, {free_size}))",
            inverse.join(", ")
        ),
        log_det: sum(determinants.into_iter()),
        size,
        free_size,
        scalar: false,
    })
}

impl Domain {
    /// A set of arrays rather than scalars: a (restored) nested power, a §03
    /// vector set, or the fork's `corrcholesky(n)`.
    fn is_array_set(&self) -> bool {
        matches!(
            self,
            Self::CartPow { .. }
                | Self::StdSimplex { .. }
                | Self::Ordered { .. }
                | Self::ZeroSum { .. }
                | Self::CorrCholesky { .. }
        )
    }

    fn has_complex_elements(&self) -> bool {
        match self {
            Self::Complexes => true,
            Self::CartPow { of, .. } => of.has_complex_elements(),
            Self::CartProd { of } => of.iter().any(Self::has_complex_elements),
            _ => false,
        }
    }

    pub(crate) fn chart_with_correlation_factors(
        &self,
        point: &str,
        correlations: &[String],
    ) -> Result<Chart, String> {
        if correlations.is_empty() {
            return self.chart(point);
        }
        let fields = record_fields(self).ok_or("correlation factors require a record input")?;
        record_chart(&fields, point, correlations)
    }

    pub(crate) fn chart(&self, point: &str) -> Result<Chart, String> {
        let scalar =
            |domain: String, constrain: String, unconstrain: String, log_det: String| Chart {
                domain,
                free: "reals".into(),
                constrain,
                unconstrain,
                log_det,
                size: 1,
                free_size: 1,
                scalar: true,
            };
        let positive_log_det =
            || format!("logdensityof(pushfwd(log, Lebesgue(posreals)), {point})");
        match self {
            Self::Reals => Ok(scalar(
                "reals".into(),
                point.into(),
                point.into(),
                "0.0".into(),
            )),
            Self::Complexes => Ok(Chart {
                domain: "complexes".into(),
                free: "cartpow(reals, 2)".into(),
                constrain: format!("complex({point}[1], {point}[2])"),
                unconstrain: format!("[real({point}), imag({point})]"),
                log_det: "0.0".into(),
                size: 1,
                free_size: 2,
                scalar: true,
            }),
            Self::PosReals | Self::NonNegReals => Ok(scalar(
                if matches!(self, Self::PosReals) {
                    "posreals"
                } else {
                    "nonnegreals"
                }
                .into(),
                format!("exp({point})"),
                format!("log({point})"),
                positive_log_det(),
            )),
            Self::UnitInterval => Self::Interval {
                lo: crate::Bound(0.0),
                hi: crate::Bound(1.0),
            }
            .chart(point),
            Self::Interval { lo, hi } => {
                let (lo, hi) = (lo.0, hi.0);
                if lo.is_nan() || hi.is_nan() || lo >= hi {
                    return Err("a coordinate transform needs a nonempty interval".into());
                }
                let domain = format!("interval({lo}, {hi})");
                let (forward, inverse, log_det) = match (lo.is_finite(), hi.is_finite()) {
                    (false, false) => (point.into(), point.into(), "0.0".into()),
                    (true, false) => (
                        format!("{lo} + exp({point})"),
                        format!("log({point} - ({lo}))"),
                        positive_log_det(),
                    ),
                    (false, true) => (
                        format!("{hi} - exp({point})"),
                        format!("log(({hi}) - {point})"),
                        positive_log_det(),
                    ),
                    (true, true) => {
                        let width = hi - lo;
                        if !width.is_finite() {
                            return Err("interval width overflows the coordinate transform".into());
                        }
                        (
                            format!("{lo} + ({width}) * invlogit({point})"),
                            format!("logit(({point} - ({lo})) / ({width}))"),
                            format!(
                                "log({width}) + logdensityof(pushfwd(logit, Lebesgue(unitinterval)), {point})"
                            ),
                        )
                    }
                };
                Ok(scalar(domain, forward, inverse, log_det))
            }
            Self::CartPow { of, shape } => {
                let shape = shape
                    .iter()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                    .ok_or("bind domain sizes before building coordinate transforms")?;
                let size = shape
                    .iter()
                    .try_fold(1u32, |a, &b| a.checked_mul(b))
                    .filter(|&n| n > 0)
                    .ok_or("coordinate array size is zero or overflows")?;
                if let [rows] = shape.as_slice()
                    && let Some(fields) = record_fields(of)
                {
                    let var = fresh_row(point);
                    let row = record_chart(&fields, &var, &[])?;
                    let map_rows = |expr: &str| format!("broadcast({var} -> {expr}, {point})");
                    return Ok(Chart {
                        domain: format!("cartpow({}, {rows})", row.domain),
                        free: format!("cartpow({}, {rows})", row.free),
                        constrain: map_rows(&row.constrain),
                        unconstrain: map_rows(&row.unconstrain),
                        log_det: if row.log_det == "0.0" {
                            "0.0".into()
                        } else {
                            format!("sum({})", map_rows(&row.log_det))
                        },
                        size: 0,
                        free_size: 0,
                        scalar: false,
                    });
                }
                let dims = shape
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                // A power of arrays (a nested array, vectors of a §03 vector
                // set, correlation factors): each cell takes the element's
                // chart. The free value is the power of the element's free
                // set, so it stays nested in the same way, and the log volumes
                // of the cells add.
                if of.is_array_set() {
                    let var = fresh_row(point);
                    let cell = of.chart(&var)?;
                    let map = |expr: &str| {
                        if expr == var {
                            point.to_string()
                        } else {
                            format!("broadcast({var} -> {expr}, {point})")
                        }
                    };
                    return Ok(Chart {
                        domain: format!("cartpow({}, [{dims}])", cell.domain),
                        free: format!("cartpow({}, [{dims}])", cell.free),
                        constrain: map(&cell.constrain),
                        unconstrain: map(&cell.unconstrain),
                        log_det: if cell.log_det == "0.0" {
                            "0.0".into()
                        } else {
                            format!("sum({})", map(&cell.log_det))
                        },
                        // Nested cells do not flatten into a positional product.
                        size: 0,
                        free_size: 0,
                        scalar: false,
                    });
                }
                if matches!(of.as_ref(), Self::Complexes) {
                    let rank = shape.len();
                    let axes = ":, ".repeat(rank);
                    // Each element contributes one last-axis block [real, imag].
                    return Ok(Chart {
                        domain: format!("cartpow(complexes, [{dims}])"),
                        free: format!("cartpow(reals, [{dims}, 2])"),
                        constrain: format!("complex.({point}[{axes}1], {point}[{axes}2])"),
                        unconstrain: format!(
                            "joinblocks(broadcast(z -> addaxes([real(z), imag(z)], {rank}, 0), addaxes({point}, 0, 1)))"
                        ),
                        log_det: "0.0".into(),
                        size,
                        free_size: size.checked_mul(2).ok_or("coordinate size overflows")?,
                        scalar: false,
                    });
                }
                let cell = of.chart("_")?;
                if !cell.scalar {
                    return Err(
                        "Cartesian powers of non-scalar domains need a structured chart".into(),
                    );
                }
                let lift = |expression: &str| {
                    if expression == "_" {
                        point.into()
                    } else {
                        format!("broadcast(fn({expression}), {point})")
                    }
                };
                Ok(Chart {
                    domain: format!("cartpow({}, [{dims}])", cell.domain),
                    free: format!("cartpow(reals, [{dims}])"),
                    constrain: lift(&cell.constrain),
                    unconstrain: lift(&cell.unconstrain),
                    log_det: if cell.log_det == "0.0" {
                        "0.0".into()
                    } else {
                        format!("sum({})", lift(&cell.log_det))
                    },
                    size,
                    free_size: size,
                    scalar: false,
                })
            }
            Self::Record { fields } => record_chart(fields, point, &[]),
            Self::Ordered { n: Some(n) } if *n > 0 => {
                let (forward, inverse, log_det) = if *n == 1 {
                    (point.into(), point.into(), "0.0".into())
                } else {
                    let tail = vector_range(point, 2, *n);
                    let head = vector_range(point, 1, n - 1);
                    (
                        format!("cumsum(cat([{point}[1]], exp.({tail})))"),
                        format!("cat([{point}[1]], log.({tail} .- {head}))"),
                        format!("sum({tail})"),
                    )
                };
                Ok(Chart {
                    domain: format!("ordered({n})"),
                    free: format!("cartpow(reals, {n})"),
                    constrain: forward,
                    unconstrain: inverse,
                    log_det,
                    size: *n,
                    free_size: *n,
                    scalar: false,
                })
            }
            Self::ZeroSum { n: Some(n) } if *n > 1 => Ok(Chart {
                domain: format!("zerosum({n})"),
                free: format!("cartpow(reals, {})", n - 1),
                constrain: format!("cat({point}, [-sum({point})])"),
                unconstrain: vector_range(point, 1, n - 1),
                log_det: "0.0".into(),
                size: *n,
                free_size: n - 1,
                scalar: false,
            }),
            // `corrcholesky(1)` is the single point [[1]], with no free coordinate.
            Self::CorrCholesky { n: Some(n) } if *n > 1 => {
                cholesky_factor_chart(*n, format!("corrcholesky({n})"), point)
            }
            Self::StdSimplex { n: Some(n) } if *n > 1 => {
                let offsets = (1..*n)
                    .rev()
                    .map(|i| (i as f64).ln().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let z = format!("({point} .- [{offsets}])");
                let breaks = format!("invlogit.({z})");
                let log_sig =
                    |x: &str| format!("((-abs.({x}) .+ {x}) ./ 2.0 .- log1p.(exp.(-abs.({x}))))");
                // Prefix sums in log space avoid products that round to zero
                // before their derivatives, and use the qualified cumsum AD path.
                let remaining = format!("exp.(cumsum({}))", log_sig(&format!("(-{z})")));
                let before = if *n == 2 {
                    "[1.0]".into()
                } else {
                    format!(
                        "cat([1.0], {})",
                        vector_range(&format!("({remaining})"), 1, n - 2)
                    )
                };
                let head = vector_range(point, 1, n - 1);
                // Sum positive tails rather than subtract prefixes from one:
                // a valid interior point can have a prefix that rounds to one.
                let tail = vector_select(point, (2..=*n).rev());
                let remaining_tail = vector_select(&format!("(cumsum({tail}))"), (1..*n).rev());
                // Centered stick-breaking. The reference is coordinate Lebesgue,
                // so there is no surface-volume factor sqrt(n).
                let counts = (1..*n)
                    .rev()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(Chart {
                    domain: format!("stdsimplex({n})"),
                    free: format!("cartpow(reals, {})", n - 1),
                    constrain: format!("cat({breaks} .* {before}, [({remaining})[{}]])", n - 1),
                    unconstrain: format!("log.({head}) .- log.({remaining_tail}) .+ [{offsets}]"),
                    log_det: format!(
                        "sum({} .+ [{counts}] .* {})",
                        log_sig(&z),
                        log_sig(&format!("(-{z})"))
                    ),
                    size: *n,
                    free_size: n - 1,
                    scalar: false,
                })
            }
            Self::CartProd { of } => {
                if let Some(fields) = record_fields(self) {
                    return Self::Record { fields }.chart(point);
                }
                let mut start = 1u32;
                let mut free_start = 1u32;
                let mut parts = Vec::new();
                let complex = of.iter().any(Self::has_complex_elements);
                for domain in of {
                    if matches!(domain, Self::CartPow { shape, of } if shape.len() != 1 || of.is_array_set())
                        || matches!(domain, Self::CorrCholesky { .. })
                    {
                        return Err("coordinate products currently require scalar or flat-vector components".into());
                    }
                    let base = domain.chart(point)?;
                    let end = start
                        .checked_add(base.size)
                        .ok_or("coordinate size overflows")?;
                    let free_end = free_start
                        .checked_add(base.free_size)
                        .ok_or("coordinate size overflows")?;
                    let access = |start: u32, size: u32, scalar: bool| {
                        if scalar {
                            format!("{point}[{start}]")
                        } else {
                            vector_range(point, start, start + size - 1)
                        }
                    };
                    let mut u = access(
                        free_start,
                        base.free_size,
                        base.scalar && base.free_size == 1,
                    );
                    let mut x = access(start, base.size, base.scalar);
                    // A positional product promotes real components into the complex vector.
                    if complex && !domain.has_complex_elements() {
                        x = if base.scalar {
                            format!("real({x})")
                        } else {
                            format!("real.({x})")
                        };
                    }
                    let complex_vector = matches!(domain, Self::CartPow { of, .. } if matches!(of.as_ref(), Self::Complexes));
                    if complex_vector {
                        u = format!("array({u}, [{}, 2], [1, 2])", base.size);
                    }
                    let forward = domain.chart(&u)?;
                    let mut inverse = domain.chart(&x)?;
                    if complex_vector {
                        inverse.unconstrain =
                            format!("joinblocks(broadcast(z -> [real(z), imag(z)], {x}))");
                    }
                    start = end;
                    free_start = free_end;
                    parts.push((base, forward, inverse));
                }
                let join = |value: fn(&(Chart, Chart, Chart)) -> String| {
                    parts.iter().map(value).collect::<Vec<_>>().join(", ")
                };
                let array =
                    |s: &str, scalar: bool| if scalar { format!("[{s}]") } else { s.into() };
                Ok(Chart {
                    domain: format!("cartprod({})", join(|p| p.0.domain.clone())),
                    free: if complex {
                        format!("cartpow(reals, {})", free_start - 1)
                    } else {
                        format!("cartprod({})", join(|p| p.0.free.clone()))
                    },
                    constrain: format!(
                        "cat({})",
                        parts
                            .iter()
                            .map(|p| array(&p.1.constrain, p.0.scalar))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    unconstrain: format!(
                        "cat({})",
                        parts
                            .iter()
                            .map(|p| array(&p.2.unconstrain, p.0.scalar && p.0.free_size == 1))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    log_det: sum(parts.iter().map(|p| p.1.log_det.clone())),
                    size: start - 1,
                    free_size: free_start - 1,
                    scalar: false,
                })
            }
            _ => Err(format!("no continuous coordinate transform for {self:?}")),
        }
    }
}
