//! Packed literal vectors: fixed data held as one typed buffer.
//!
//! A [`Node::Packed`](crate::Node::Packed) has the value of `(vector lit_1 … lit_n)`
//! over its elements, so every reader treats the two spellings alike through
//! [`Module::literals`](crate::Module::literals). Fixed `load_data` arrays take the
//! packed form: an array costs one node plus eight bytes per element instead of
//! one literal node, its side-table entries, and its emitted copies per element.

use std::sync::Arc;

use crate::{Idx, Module, Node, NodeId, Scalar, ScalarType};

/// A nonempty rank-1 vector of literal scalars of one kind, in one shared
/// buffer. Cloning shares the buffer, so module clones stay cheap.
#[derive(Clone, Debug)]
pub enum Packed {
    Real(Arc<[f64]>),
    Int(Arc<[i64]>),
    Bool(Arc<[bool]>),
}

impl Packed {
    /// The shortest vector producers pack. A one-element `vector(x)` keeps
    /// its call spelling: passes read such singleton wrappers by their
    /// element node.
    pub const MIN_LEN: usize = 2;

    /// Pack scalars of one numeric or boolean kind. `None` for an empty list, a
    /// string, or mixed kinds: those keep the `vector` call spelling.
    pub fn from_scalars<'a>(values: impl IntoIterator<Item = &'a Scalar>) -> Option<Self> {
        let mut values = values.into_iter().peekable();
        let packed = match values.peek()? {
            Scalar::Real(_) => Self::Real(
                values
                    .map(|v| match v {
                        Scalar::Real(x) => Some(*x),
                        _ => None,
                    })
                    .collect::<Option<_>>()?,
            ),
            Scalar::Int(_) => Self::Int(
                values
                    .map(|v| match v {
                        Scalar::Int(x) => Some(*x),
                        _ => None,
                    })
                    .collect::<Option<_>>()?,
            ),
            Scalar::Bool(_) => Self::Bool(
                values
                    .map(|v| match v {
                        Scalar::Bool(x) => Some(*x),
                        _ => None,
                    })
                    .collect::<Option<_>>()?,
            ),
            Scalar::Str(_) => return None,
        };
        Some(packed)
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Real(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    /// Always `false` for a packed vector built by [`Packed::from_scalars`].
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element kind every entry shares.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::Real(_) => ScalarType::Real,
            Self::Int(_) => ScalarType::Integer,
            Self::Bool(_) => ScalarType::Boolean,
        }
    }

    /// Element `i` (zero-based) as a literal scalar.
    pub fn get(&self, i: usize) -> Option<Scalar> {
        match self {
            Self::Real(v) => v.get(i).map(|&x| Scalar::Real(x)),
            Self::Int(v) => v.get(i).map(|&x| Scalar::Int(x)),
            Self::Bool(v) => v.get(i).map(|&x| Scalar::Bool(x)),
        }
    }

    /// Every element in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Scalar> + '_ {
        (0..self.len()).map(|i| self.get(i).expect("index below len"))
    }

    /// The elements at `indices` (zero-based), in that order. `None` when an
    /// index is out of range or the selection is empty.
    pub fn select(&self, indices: impl IntoIterator<Item = usize>) -> Option<Self> {
        fn pick<T: Copy>(v: &[T], indices: impl IntoIterator<Item = usize>) -> Option<Arc<[T]>> {
            let picked = indices
                .into_iter()
                .map(|i| v.get(i).copied())
                .collect::<Option<Arc<[T]>>>()?;
            (!picked.is_empty()).then_some(picked)
        }
        Some(match self {
            Self::Real(v) => Self::Real(pick(v, indices)?),
            Self::Int(v) => Self::Int(pick(v, indices)?),
            Self::Bool(v) => Self::Bool(pick(v, indices)?),
        })
    }

    /// Do both share one buffer?
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Real(a), Self::Real(b)) => Arc::ptr_eq(a, b),
            (Self::Int(a), Self::Int(b)) => Arc::ptr_eq(a, b),
            (Self::Bool(a), Self::Bool(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Equal kinds and equal elements, as the `vector` call spellings would compare.
impl PartialEq for Packed {
    fn eq(&self, other: &Self) -> bool {
        if self.ptr_eq(other) {
            // A shared buffer still compares NaN entries unequal, like `Scalar`.
            return !matches!(self, Self::Real(v) if v.iter().any(|x| x.is_nan()));
        }
        match (self, other) {
            (Self::Real(a), Self::Real(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}

/// The elements of a literal vector in either spelling: a `vector` call whose
/// arguments are all literal nodes, or a packed vector.
#[derive(Clone, Copy, Debug)]
pub enum Literals<'a> {
    /// `(vector lit …)`; every argument is a [`Node::Lit`].
    Nodes(&'a Module, &'a [NodeId]),
    Packed(&'a Packed),
}

impl<'a> Literals<'a> {
    pub fn len(&self) -> usize {
        match self {
            Self::Nodes(_, args) => args.len(),
            Self::Packed(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element `i` (zero-based).
    pub fn get(&self, i: usize) -> Option<Scalar> {
        match self {
            Self::Nodes(m, args) => match m.node(*args.get(i)?) {
                Node::Lit(s) => Some(s.clone()),
                _ => None,
            },
            Self::Packed(p) => p.get(i),
        }
    }

    /// Every element in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Scalar> + 'a {
        let this = *self;
        (0..this.len()).map(move |i| this.get(i).expect("literal element"))
    }

    /// Every element as an integer, or `None` if one is not an integer literal.
    pub fn ints(&self) -> Option<Vec<i64>> {
        match self {
            Self::Packed(Packed::Int(v)) => Some(v.to_vec()),
            Self::Packed(_) => None,
            Self::Nodes(..) => self
                .iter()
                .map(|s| match s {
                    Scalar::Int(i) => Some(i),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Every element as a real number: integer and boolean literals convert.
    /// `None` if one is a string.
    pub fn reals(&self) -> Option<Vec<f64>> {
        match self {
            Self::Packed(Packed::Real(v)) => Some(v.to_vec()),
            _ => self
                .iter()
                .map(|s| match s {
                    Scalar::Real(x) => Some(x),
                    Scalar::Int(i) => Some(i as f64),
                    Scalar::Bool(b) => Some(f64::from(u8::from(b))),
                    Scalar::Str(_) => None,
                })
                .collect(),
        }
    }

    /// The element kind every entry shares, if any.
    pub fn scalar_type(&self) -> Option<ScalarType> {
        match self {
            Self::Packed(p) => Some(p.scalar_type()),
            Self::Nodes(..) => {
                let mut kinds = self.iter().map(|s| match s {
                    Scalar::Real(_) => Some(ScalarType::Real),
                    Scalar::Int(_) => Some(ScalarType::Integer),
                    Scalar::Bool(_) => Some(ScalarType::Boolean),
                    Scalar::Str(_) => None,
                });
                let first = kinds.next()??;
                kinds.all(|k| k == Some(first)).then_some(first)
            }
        }
    }
}

impl Module {
    /// The elements of `id` when it is a literal vector in either spelling: a
    /// packed vector, or a `vector` call whose arguments are all literal
    /// nodes. References are not followed.
    pub fn literals(&self, id: NodeId) -> Option<Literals<'_>> {
        match self.node(id) {
            Node::Packed(p) => Some(Literals::Packed(p)),
            Node::Call(c)
                if c.inputs.is_none()
                    && c.named.is_empty()
                    && matches!(c.head, crate::CallHead::Builtin(h) if self.resolve(h) == "vector")
                    && c.args.iter().all(|&a| matches!(self.node(a), Node::Lit(_))) =>
            {
                Some(Literals::Nodes(self, &c.args))
            }
            _ => None,
        }
    }

    /// The node for element `i` (zero-based) of a literal vector: the literal
    /// argument itself, or for a packed vector a new literal annotated as
    /// inference annotates one. For passes that rebuild an expression from a
    /// few selected elements; never call it per element of a large vector.
    pub fn literal_element(&mut self, vector: NodeId, i: usize) -> Option<NodeId> {
        match self.node(vector) {
            Node::Packed(p) => {
                let value = p.get(i)?;
                let ty = crate::Type::Scalar(p.scalar_type());
                let set = crate::ValueSet::of_literal(&value);
                let id = self.alloc(Node::Lit(value));
                self.set_type(id, ty);
                self.set_phase(id, crate::Phase::Fixed);
                self.set_valueset(id, set);
                Some(id)
            }
            Node::Call(c) => c.args.get(i).copied(),
            _ => None,
        }
    }

    /// Every element node of a `vector` call, or of a packed vector through
    /// [`Module::literal_element`]. For short structural vectors: bin edges,
    /// coefficients, selectors.
    pub fn vector_elements(&mut self, vector: NodeId) -> Option<Vec<NodeId>> {
        match self.node(vector) {
            Node::Packed(p) => {
                let n = p.len();
                (0..n).map(|i| self.literal_element(vector, i)).collect()
            }
            Node::Call(c) if matches!(c.head, crate::CallHead::Builtin(h) if self.resolve(h) == "vector") => {
                Some(c.args.to_vec())
            }
            _ => None,
        }
    }

    /// Respell every `vector` call of at least [`Packed::MIN_LEN`] same-kind
    /// numeric or boolean literals as a packed vector, in place, and return
    /// how many changed. Values, types,
    /// spans and references are unchanged; the old literal children become
    /// unreachable.
    pub fn pack_literal_vectors(&mut self) -> usize {
        let mut packed = 0;
        for i in 0..self.node_count() {
            let id = NodeId::from_usize(i);
            let Some(Literals::Nodes(_, args)) = self.literals(id) else {
                continue;
            };
            if args.len() < Packed::MIN_LEN {
                continue;
            }
            let Node::Call(c) = self.node(id) else {
                unreachable!()
            };
            let values = c.args.iter().map(|&a| match self.node(a) {
                Node::Lit(s) => s,
                _ => unreachable!(),
            });
            if let Some(p) = Packed::from_scalars(values) {
                self.respell_node(id, Node::Packed(p));
                packed += 1;
            }
        }
        packed
    }

    /// A literal vector of `values`: packed when there are at least
    /// [`Packed::MIN_LEN`] of one numeric or boolean kind, else a `vector`
    /// call of literal nodes. `None` when empty.
    pub fn alloc_literals(&mut self, values: &[Scalar]) -> Option<NodeId> {
        if values.is_empty() {
            return None;
        }
        if values.len() >= Packed::MIN_LEN
            && let Some(packed) = Packed::from_scalars(values)
        {
            return Some(self.alloc(Node::Packed(packed)));
        }
        let args: Box<[NodeId]> = values
            .iter()
            .map(|v| self.alloc(Node::Lit(v.clone())))
            .collect();
        let head = crate::CallHead::Builtin(self.intern("vector"));
        Some(self.alloc(Node::Call(crate::Call {
            head,
            args,
            named: Box::new([]),
            inputs: None,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_needs_one_numeric_or_boolean_kind() {
        let reals = [Scalar::Real(1.5), Scalar::Real(-2.0)];
        assert_eq!(
            Packed::from_scalars(&reals),
            Some(Packed::Real(Arc::from([1.5, -2.0])))
        );
        assert!(Packed::from_scalars(&[Scalar::Real(1.0), Scalar::Int(1)]).is_none());
        assert!(Packed::from_scalars(&[Scalar::Str("a".into())]).is_none());
        assert!(Packed::from_scalars(&[]).is_none());
    }

    #[test]
    fn both_spellings_read_the_same_elements() {
        let mut m = Module::default();
        let values = [Scalar::Int(3), Scalar::Int(1), Scalar::Int(2)];
        let packed = m.alloc_literals(&values).unwrap();
        let args: Box<[NodeId]> = values
            .iter()
            .map(|v| m.alloc(Node::Lit(v.clone())))
            .collect();
        let head = crate::CallHead::Builtin(m.intern("vector"));
        let call = m.alloc(Node::Call(crate::Call {
            head,
            args,
            named: Box::new([]),
            inputs: None,
        }));
        for id in [packed, call] {
            let lits = m.literals(id).unwrap();
            assert_eq!(lits.len(), 3);
            assert_eq!(lits.ints(), Some(vec![3, 1, 2]));
            assert_eq!(lits.reals(), Some(vec![3.0, 1.0, 2.0]));
            assert_eq!(lits.scalar_type(), Some(ScalarType::Integer));
            assert_eq!(lits.iter().collect::<Vec<_>>(), values.to_vec());
        }
    }

    #[test]
    fn a_shared_buffer_with_nan_is_unequal_like_its_literals() {
        let p = Packed::Real(Arc::from([f64::NAN]));
        assert_ne!(p, p.clone());
        let q = Packed::Real(Arc::from([1.0]));
        assert_eq!(q, q.clone());
        assert_eq!(q.select([0, 0]), Some(Packed::Real(Arc::from([1.0, 1.0]))));
        assert_eq!(q.select([1]), None);
    }
}
