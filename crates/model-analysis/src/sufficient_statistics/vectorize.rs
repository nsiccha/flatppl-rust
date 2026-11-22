//! Sum same-shape scalar group terms as one elementwise term.
//!
//! A fixed partition scores every occupied group with the same family
//! expression over its own summaries and parameter cell. Added one by one,
//! the scoring program grows with the number of groups, and every term costs
//! its own kernels on each call. Terms of one shape become one `broadcast`
//! over stacked group vectors and a `sum`: their varying literals, summaries
//! and parameter cells stack into vectors. The summaries themselves, and so
//! the preparation ABI, stay unchanged. Terms that do not unify keep their
//! scalar form.

use std::collections::HashMap;

use flatppl_core::graph::{build_call, fold_add};
use flatppl_core::{Call, CallHead, Idx, Module, NamedArg, NamedKind, Node, NodeId, Scalar, Type};

/// Sum `terms` (at least one), vectorizing each class of same-shape terms.
pub(super) fn sum_terms(m: &mut Module, terms: &[NodeId]) -> NodeId {
    if terms.len() < 2 {
        return fold_add(m, terms);
    }
    // Generated summaries and terms carry no types yet, and terms are not yet
    // bound. The elementwise proof needs both.
    let _ = flatppl_infer::infer(m);
    for &term in terms {
        flatppl_infer::annotate_rebuilt(m, term);
    }
    let mut shapes = HashMap::new();
    let mut classes: Vec<(String, Vec<NodeId>)> = Vec::new();
    for &term in terms {
        let key = shape(m, term, &mut shapes);
        match classes.iter_mut().find(|(k, _)| *k == key) {
            Some((_, members)) => members.push(term),
            None => classes.push((key, vec![term])),
        }
    }
    let mut sums = Vec::new();
    let mut vectorized = Vec::new();
    for (_, members) in classes {
        if members.len() > 1 && scalar(m, members[0]) {
            let mut unifier = Unifier::default();
            if let Some((vector, true)) = unifier.unify(m, &members) {
                let sum = build_call(m, "sum", &[vector]);
                vectorized.push((sum, members, unifier.created));
                continue;
            }
        }
        sums.extend(members);
    }
    if !vectorized.is_empty() {
        // Keep a class's vectorized form only if inference types every node it
        // added; otherwise its scalar terms stay.
        let _ = flatppl_infer::infer(m);
        for (sum, members, created) in vectorized {
            flatppl_infer::annotate_rebuilt(m, sum);
            if scalar(m, sum) && created.iter().all(|&id| typed(m, id)) {
                sums.push(sum);
            } else {
                sums.extend(members);
            }
        }
    }
    fold_add(m, &sums)
}

fn typed(m: &Module, id: NodeId) -> bool {
    matches!(m.type_of(id), Some(ty) if !matches!(ty, Type::Deferred | Type::Any | Type::Failed(_)))
}

/// A key equal for terms that differ only in literal values and references.
fn shape(m: &Module, id: NodeId, memo: &mut HashMap<NodeId, String>) -> String {
    if let Some(key) = memo.get(&id) {
        return key.clone();
    }
    let key = match m.node(id) {
        Node::Lit(Scalar::Real(_)) => "r".into(),
        Node::Lit(Scalar::Int(_)) => "i".into(),
        Node::Lit(Scalar::Bool(_)) => "b".into(),
        Node::Lit(Scalar::Str(s)) => format!("s{s:?}"),
        Node::Const(symbol) => format!("c{}", m.resolve(*symbol)),
        Node::Ref(_) => "ref".into(),
        Node::Call(call) if call.inputs.is_none() => match call.head {
            CallHead::Builtin(head) => {
                let mut key = format!("{}(", m.resolve(head));
                for &arg in &call.args {
                    key += &shape(m, arg, memo);
                    key.push(',');
                }
                for arg in &call.named {
                    key += &format!("{:?}{}=", arg.kind, m.resolve(arg.name));
                    key += &shape(m, arg.value, memo);
                    key.push(',');
                }
                key.push(')');
                key
            }
            CallHead::User(_) => format!("#{}", id.index()),
        },
        _ => format!("#{}", id.index()),
    };
    memo.insert(id, key.clone());
    key
}

fn scalar(m: &Module, id: NodeId) -> bool {
    match m.node(id) {
        Node::Lit(Scalar::Real(_) | Scalar::Int(_) | Scalar::Bool(_)) => true,
        _ => matches!(m.type_of(id), Some(Type::Scalar(_))),
    }
}

/// A broadcast argument shared by every group: a scalar, or a constant such
/// as a distribution constructor, so broadcasting passes it whole.
fn shareable(m: &Module, id: NodeId) -> bool {
    matches!(m.node(id), Node::Const(_)) || scalar(m, id)
}

/// Elementwise results: scalars, or records of scalars (distribution parameters).
fn elementwise(m: &Module, id: NodeId) -> bool {
    scalar(m, id)
        || matches!(m.type_of(id), Some(Type::Record(fields))
            if fields.iter().all(|(_, ty)| matches!(ty, Type::Scalar(_))))
}

#[derive(Default)]
struct Unifier {
    memo: HashMap<Vec<NodeId>, Option<(NodeId, bool)>>,
    /// The values this unifier added (not callee constants), for the typing check.
    created: Vec<NodeId>,
}

impl Unifier {
    /// One node per column of group nodes, and whether it varies by group
    /// (then it holds one element per group, in member order).
    fn unify(&mut self, m: &mut Module, ids: &[NodeId]) -> Option<(NodeId, bool)> {
        if let Some(done) = self.memo.get(ids) {
            return *done;
        }
        let result = self.unify_new(m, ids);
        self.memo.insert(ids.to_vec(), result);
        result
    }

    fn unify_new(&mut self, m: &mut Module, ids: &[NodeId]) -> Option<(NodeId, bool)> {
        let first = ids[0];
        if ids[1..].iter().all(|&id| m.structural_eq(first, id)) {
            return Some((first, false));
        }
        if let Node::Call(call) = m.node(first).clone() {
            let CallHead::Builtin(head) = call.head else {
                return None;
            };
            let calls = ids
                .iter()
                .map(|&id| match m.node(id) {
                    Node::Call(c)
                        if c.inputs.is_none()
                            && c.head == call.head
                            && c.args.len() == call.args.len()
                            && c.named.len() == call.named.len()
                            && c.named
                                .iter()
                                .zip(&call.named)
                                .all(|(a, b)| a.kind == b.kind && a.name == b.name) =>
                    {
                        Some(c.clone())
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            if m.resolve(head) == "get" && call.args.len() == 2 && call.named.is_empty() {
                return self.gather(m, &calls);
            }
            if !ids.iter().all(|&id| elementwise(m, id)) {
                return None;
            }
            let mut args = vec![m.alloc(Node::Const(head))];
            for j in 0..call.args.len() {
                let column = calls.iter().map(|c| c.args[j]).collect::<Vec<_>>();
                let (arg, varying) = self.unify(m, &column)?;
                if !varying && !shareable(m, arg) {
                    return None;
                }
                args.push(arg);
            }
            // Broadcast passes names as keywords: `broadcast(record, mu = …)`
            // is the elementwise form of `record(mu = …)`.
            let record = m.resolve(head) == "record";
            let mut named = Vec::with_capacity(call.named.len());
            for (j, template) in call.named.iter().enumerate() {
                if !(template.kind == NamedKind::Kwarg
                    || (record && template.kind == NamedKind::Field))
                {
                    return None;
                }
                let column = calls.iter().map(|c| c.named[j].value).collect::<Vec<_>>();
                let (value, varying) = self.unify(m, &column)?;
                if !varying && !shareable(m, value) {
                    return None;
                }
                named.push(NamedArg {
                    kind: NamedKind::Kwarg,
                    name: template.name,
                    value,
                });
            }
            let broadcast = m.intern("broadcast");
            let node = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(broadcast),
                args: args.into(),
                named: named.into(),
                inputs: None,
            }));
            self.created.push(node);
            return Some((node, true));
        }
        if !ids.iter().all(|&id| scalar(m, id)) {
            return None;
        }
        let stacked = build_call(m, "vector", ids);
        self.created.push(stacked);
        Some((stacked, true))
    }

    /// `get(array, k)` with a shared array and a literal cell per group gathers once.
    fn gather(&mut self, m: &mut Module, calls: &[Call]) -> Option<(NodeId, bool)> {
        let container = calls[0].args[0];
        if !calls.iter().all(|c| m.structural_eq(container, c.args[0])) {
            return None;
        }
        let cells = calls.iter().map(|c| c.args[1]).collect::<Vec<_>>();
        if !cells
            .iter()
            .all(|&cell| matches!(m.node(cell), Node::Lit(Scalar::Int(_))))
        {
            return None;
        }
        let cells = build_call(m, "vector", &cells);
        let gathered = build_call(m, "get", &[container, cells]);
        self.created.extend([cells, gathered]);
        Some((gathered, true))
    }
}
