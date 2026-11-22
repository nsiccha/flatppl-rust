//! Fold fixed count tables before emission, including flat arrays split into rows.

use super::super::histogram::fixed_scalar;
use super::{Expr, Module, NodeId, Reducer, Scalar, call, num, op, resolve, vector};
use flatppl_core::Node;

pub(super) struct Summaries {
    center: Vec<i64>,
    valid: i64,
    counts: Vec<i64>,
    factorials: f64,
}

impl Summaries {
    pub(super) fn center(&self) -> Expr {
        vector(self.center.iter().copied().map(Expr::Integer))
    }

    pub(super) fn counts(&self) -> Expr {
        vector(self.counts.iter().copied().map(Expr::Integer))
    }

    pub(super) fn valid(&self) -> Expr {
        Expr::Integer(self.valid)
    }

    pub(super) fn factorials(&self) -> Expr {
        if self.factorials < 0. {
            op("neg", [num(-self.factorials)])
        } else {
            num(self.factorials)
        }
    }
}

pub(super) fn summaries(
    reducer: &Reducer,
    m: &Module,
    observations: NodeId,
    rows: usize,
    width: usize,
    trials: i64,
) -> Option<Summaries> {
    let source = resolve(m, observations, &reducer.boundary);
    if call(m, source, "vector").is_none() && call(m, source, "splitblocks").is_none() {
        return None;
    }
    // A row or the cell vector may be packed; its entries read as their literals.
    let counts = |id: NodeId| -> Option<Vec<i64>> {
        let id = resolve(m, id, &reducer.boundary);
        match m.node(id) {
            Node::Packed(flatppl_core::Packed::Int(v)) => Some(v.to_vec()),
            Node::Packed(_) => None,
            _ => call(m, id, "vector")?
                .args
                .iter()
                .map(|&cell| match fixed_scalar(m, cell, reducer)? {
                    Scalar::Int(k) => Some(k),
                    _ => None,
                })
                .collect(),
        }
    };
    let mut result = Summaries {
        center: vec![0; width],
        valid: 0,
        counts: vec![0; width],
        factorials: 0.,
    };
    // The caller bounds trials to 63 and the total count to the target integer range.
    let mut frequency = vec![0i64; trials as usize + 1];
    let mut row = Vec::with_capacity(width);
    let mut add = |cells: &[i64]| {
        if cells.len() != width {
            return None;
        }
        row.clear();
        row.extend_from_slice(cells);
        if row.iter().all(|k| (0..=trials).contains(k)) && row.iter().sum::<i64>() == trials {
            if result.valid == 0 {
                result.center.copy_from_slice(&row);
            }
            result.valid += 1;
            for (total, &k) in result.counts.iter_mut().zip(&row) {
                *total += k;
                frequency[k as usize] += 1;
            }
        }
        Some(())
    };
    if let Some(table) = call(m, source, "vector") {
        if table.args.len() != rows {
            return None;
        }
        for &item in &table.args {
            add(&counts(item)?)?;
        }
    } else {
        let [data, block] = *call(m, source, "splitblocks")?.args else {
            return None;
        };
        if fixed_scalar(m, block, reducer)? != Scalar::Int(width as i64) {
            return None;
        }
        let cells = counts(data)?;
        if cells.len() != rows.checked_mul(width)? {
            return None;
        }
        for cells in cells.chunks_exact(width) {
            add(cells)?;
        }
    }
    // Subtract the reference row's factorial multiplicities before taking logs.
    for &k in &result.center {
        frequency[k as usize] -= result.valid;
    }
    let mut log_factorial = 0.;
    for (k, times) in frequency.into_iter().enumerate().skip(2) {
        log_factorial += (k as f64).ln();
        result.factorials += times as f64 * log_factorial;
    }
    Some(result)
}
