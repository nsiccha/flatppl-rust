//! Fixed host values become ordinary FlatPPL literal and constructor nodes, and
//! literal bindings read back as host values.

use flatppl_core::{
    Call, CallHead, Dim, Module, NamedArg, NamedKind, Node, NodeId, Packed, RefNs, Scalar,
    ScalarType, Symbol, Type, ValueSet,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

/// A fixed value. JSON writes infinite reals as `"inf"` and `"-inf"`, and an
/// array as one typed flat list, `{"shape", "dtype", "data"}`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Constant {
    Bool(bool),
    Integer(i64),
    Real(#[serde(with = "real")] f64),
    /// Preserve binary32 source precision until fixed-data validation.
    Real32(#[serde(with = "real32")] f32),
    Complex(#[serde(with = "real")] f64, #[serde(with = "real")] f64),
    String(String),
    /// Flat data in axis order from slowest to fastest (C order).
    #[serde(
        serialize_with = "packed::serialize",
        deserialize_with = "packed::deserialize"
    )]
    Array {
        shape: Vec<u32>,
        data: ArrayData,
    },
    Record(Vec<(String, Constant)>),
    Table(Vec<(String, Constant)>),
    Tuple(Vec<Constant>),
    /// A `fixed(value)` marker read from source. It binds no external.
    Fixed(Box<Constant>),
    /// A reference to another binding read from source. It binds no external.
    Reference(String),
}

/// An array's entries in C order: one shared buffer per scalar kind, so a
/// large array costs its entries' bytes. Arrays of arrays keep their entries as
/// values. Entries read back as scalar [`Constant`]s.
#[derive(Clone, Debug)]
pub enum ArrayData {
    Bool(Arc<[bool]>),
    Integer(Arc<[i64]>),
    Real(Arc<[f64]>),
    /// Preserve binary32 source precision until fixed-data validation.
    Real32(Arc<[f32]>),
    Complex(Arc<[(f64, f64)]>),
    /// Entries that are not scalars of one kind, such as the vectors of a
    /// vector of vectors.
    Values(Vec<Constant>),
}

impl ArrayData {
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::Integer(v) => v.len(),
            Self::Real(v) => v.len(),
            Self::Real32(v) => v.len(),
            Self::Complex(v) => v.len(),
            Self::Values(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entry `i` in C order.
    pub fn get(&self, i: usize) -> Option<Constant> {
        Some(match self {
            Self::Bool(v) => Constant::Bool(*v.get(i)?),
            Self::Integer(v) => Constant::Integer(*v.get(i)?),
            Self::Real(v) => Constant::Real(*v.get(i)?),
            Self::Real32(v) => Constant::Real32(*v.get(i)?),
            Self::Complex(v) => {
                let (re, im) = *v.get(i)?;
                Constant::Complex(re, im)
            }
            Self::Values(v) => v.get(i)?.clone(),
        })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Constant> + '_ {
        (0..self.len()).map(|i| self.get(i).expect("index below len"))
    }

    /// Entries `start..start + len`, shared by the view, not copied.
    fn view(&self, start: usize, len: usize) -> ArrayView<'_> {
        ArrayView {
            data: self,
            start,
            len,
        }
    }
}

/// Scalars of one kind become one buffer; anything else stays values.
impl From<Vec<Constant>> for ArrayData {
    fn from(values: Vec<Constant>) -> Self {
        fn all<T>(values: &[Constant], pick: impl Fn(&Constant) -> Option<T>) -> Option<Arc<[T]>> {
            values.iter().map(pick).collect()
        }
        let packed = match values.first() {
            Some(Constant::Bool(_)) => all(&values, |v| match v {
                Constant::Bool(b) => Some(*b),
                _ => None,
            })
            .map(Self::Bool),
            Some(Constant::Integer(_)) => all(&values, |v| match v {
                Constant::Integer(i) => Some(*i),
                _ => None,
            })
            .map(Self::Integer),
            Some(Constant::Real(_)) => all(&values, |v| match v {
                Constant::Real(x) => Some(*x),
                _ => None,
            })
            .map(Self::Real),
            Some(Constant::Real32(_)) => all(&values, |v| match v {
                Constant::Real32(x) => Some(*x),
                _ => None,
            })
            .map(Self::Real32),
            Some(Constant::Complex(..)) => all(&values, |v| match v {
                Constant::Complex(re, im) => Some((*re, *im)),
                _ => None,
            })
            .map(Self::Complex),
            _ => None,
        };
        packed.unwrap_or(Self::Values(values))
    }
}

impl FromIterator<Constant> for ArrayData {
    fn from_iter<I: IntoIterator<Item = Constant>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

/// Equal entries, whatever the storage: an empty array equals any other.
impl PartialEq for ArrayData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Integer(a), Self::Integer(b)) => a == b,
            (Self::Real(a), Self::Real(b)) => a == b,
            (Self::Real32(a), Self::Real32(b)) => a == b,
            (Self::Complex(a), Self::Complex(b)) => a == b,
            _ => self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b),
        }
    }
}

/// A contiguous run of an array's entries.
#[derive(Clone, Copy)]
struct ArrayView<'a> {
    data: &'a ArrayData,
    start: usize,
    len: usize,
}

impl<'a> ArrayView<'a> {
    fn get(&self, i: usize) -> Option<Constant> {
        (i < self.len).then(|| self.data.get(self.start + i))?
    }

    fn iter(&self) -> impl Iterator<Item = Constant> + 'a {
        let view = *self;
        (0..view.len).map(move |i| view.get(i).expect("index below len"))
    }

    fn sub(&self, start: usize, len: usize) -> Option<Self> {
        (start.checked_add(len)? <= self.len).then_some(Self {
            data: self.data,
            start: self.start + start,
            len,
        })
    }
}

mod real {
    use super::*;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Json {
        Number(f64),
        Text(String),
    }

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        match *value {
            f64::INFINITY => serializer.serialize_str("inf"),
            f64::NEG_INFINITY => serializer.serialize_str("-inf"),
            value => serializer.serialize_f64(value),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        match Json::deserialize(deserializer)? {
            Json::Number(value) => Ok(value),
            Json::Text(text) if text == "inf" => Ok(f64::INFINITY),
            Json::Text(text) if text == "-inf" => Ok(f64::NEG_INFINITY),
            Json::Text(text) => Err(serde::de::Error::custom(format!(
                "expected a number, \"inf\" or \"-inf\", got {text:?}"
            ))),
        }
    }
}

mod packed {
    use super::*;

    #[derive(Deserialize, Serialize)]
    struct Real(#[serde(with = "real")] f64);

    #[derive(Deserialize, Serialize)]
    struct Real32(#[serde(with = "real32")] f32);

    #[derive(Deserialize, Serialize)]
    struct Complex(#[serde(with = "real")] f64, #[serde(with = "real")] f64);

    #[derive(Deserialize, Serialize)]
    #[serde(tag = "dtype", content = "data", rename_all = "snake_case")]
    enum Data {
        Bool(Vec<bool>),
        Integer(Vec<i64>),
        Real(Vec<Real>),
        Real32(Vec<Real32>),
        Complex(Vec<Complex>),
    }

    #[derive(Deserialize, Serialize)]
    struct Array {
        shape: Vec<u32>,
        #[serde(flatten)]
        data: Data,
    }

    pub fn serialize<S: Serializer>(
        shape: &[u32],
        data: &ArrayData,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let packed = match data {
            ArrayData::Bool(v) => Data::Bool(v.to_vec()),
            ArrayData::Integer(v) => Data::Integer(v.to_vec()),
            ArrayData::Real(v) => Data::Real(v.iter().map(|&x| Real(x)).collect()),
            ArrayData::Real32(v) => Data::Real32(v.iter().map(|&x| Real32(x)).collect()),
            ArrayData::Complex(v) => {
                Data::Complex(v.iter().map(|&(re, im)| Complex(re, im)).collect())
            }
            ArrayData::Values(v) if v.is_empty() => Data::Bool(Vec::new()),
            ArrayData::Values(_) => {
                return Err(serde::ser::Error::custom(
                    "array elements must be scalars of one kind",
                ));
            }
        };
        Array {
            shape: shape.to_vec(),
            data: packed,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<(Vec<u32>, ArrayData), D::Error> {
        let Array { shape, data } = Array::deserialize(deserializer)?;
        let data = match data {
            Data::Bool(values) => ArrayData::Bool(values.into()),
            Data::Integer(values) => ArrayData::Integer(values.into()),
            Data::Real(values) => ArrayData::Real(values.into_iter().map(|Real(v)| v).collect()),
            Data::Real32(values) => {
                ArrayData::Real32(values.into_iter().map(|Real32(v)| v).collect())
            }
            Data::Complex(values) => {
                ArrayData::Complex(values.into_iter().map(|Complex(re, im)| (re, im)).collect())
            }
        };
        Ok((shape, data))
    }
}

mod real32 {
    use super::*;

    pub fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
        real::serialize(&f64::from(*value), serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
        real::deserialize(deserializer).map(|value| value as f32)
    }
}

impl Constant {
    pub(crate) fn validated(&self, module: &Module, declaration: NodeId) -> Result<Self, String> {
        let ty = module
            .type_of(declaration)
            .ok_or("unresolved external type")?;
        let set = module
            .valueset_of(declaration)
            .ok_or("unresolved external domain")?;
        let value = self.normalized(module, ty)?;
        if !value.in_set(module, set)? {
            return Err(format!("value is outside {}", module.display_valueset(set)));
        }
        Ok(value)
    }

    pub(crate) fn normalized(&self, module: &Module, ty: &Type) -> Result<Self, String> {
        match (self, ty) {
            (_, Type::Any) => Ok(self.clone()),
            (Self::Bool(_), Type::Scalar(ScalarType::Boolean))
            | (Self::Integer(_), Type::Scalar(ScalarType::Integer))
            | (Self::Real(_) | Self::Real32(_), Type::Scalar(ScalarType::Real))
            | (Self::Complex(_, _), Type::Scalar(ScalarType::Complex)) => Ok(self.clone()),
            (Self::Bool(value), Type::Scalar(ScalarType::Integer)) => {
                Ok(Self::Integer(i64::from(*value)))
            }
            (Self::Bool(value), Type::Scalar(ScalarType::Real)) => {
                Ok(Self::Real(f64::from(*value)))
            }
            // An integer is a real (§03), rounded as any real literal is.
            (Self::Integer(value), Type::Scalar(ScalarType::Real)) => Ok(Self::Real(*value as f64)),
            (_, Type::Scalar(ScalarType::Complex)) => self
                .numeric_value()
                .map(|re| Self::Complex(re, 0.0))
                .ok_or_else(|| "complex constants require numeric values".into()),
            (
                Self::Array { shape, data },
                Type::Array {
                    shape: expected,
                    elem,
                },
            ) if shape.len() == expected.len()
                && shape
                    .iter()
                    .zip(expected.iter())
                    .all(|(&n, &d)| d == Dim::Static(n) || d == Dim::Dynamic) =>
            {
                use ScalarType::{Boolean, Complex, Integer, Real};
                // Whole-buffer conversions for the scalar rules above.
                let data = match (data, elem.as_ref()) {
                    (data, Type::Any)
                    | (data @ ArrayData::Bool(_), Type::Scalar(Boolean))
                    | (data @ ArrayData::Integer(_), Type::Scalar(Integer))
                    | (data @ (ArrayData::Real(_) | ArrayData::Real32(_)), Type::Scalar(Real))
                    | (data @ ArrayData::Complex(_), Type::Scalar(Complex)) => data.clone(),
                    (ArrayData::Bool(v), Type::Scalar(Integer)) => {
                        ArrayData::Integer(v.iter().map(|&b| i64::from(b)).collect())
                    }
                    (ArrayData::Bool(v), Type::Scalar(Real)) => {
                        ArrayData::Real(v.iter().map(|&b| f64::from(b)).collect())
                    }
                    (ArrayData::Integer(v), Type::Scalar(Real)) => {
                        ArrayData::Real(v.iter().map(|&i| i as f64).collect())
                    }
                    (data, elem) => data
                        .iter()
                        .map(|v| v.normalized(module, elem))
                        .collect::<Result<Vec<_>, _>>()?
                        .into(),
                };
                Ok(Self::Array {
                    shape: shape.clone(),
                    data,
                })
            }
            (Self::Record(fields), Type::Record(expected)) => {
                Ok(Self::Record(normalize_fields(module, fields, expected)?))
            }
            (Self::Table(fields) | Self::Record(fields), Type::Table { columns, nrows }) => {
                let expected: Vec<_> = columns
                    .iter()
                    .map(|(name, ty)| {
                        (
                            *name,
                            match ty {
                                Type::Record(fields) => Type::Table {
                                    columns: fields.clone(),
                                    nrows: *nrows,
                                },
                                _ => Type::Array {
                                    shape: vec![*nrows].into(),
                                    elem: Box::new(ty.clone()),
                                },
                            },
                        )
                    })
                    .collect();
                Ok(Self::Table(normalize_fields(module, fields, &expected)?))
            }
            (Self::Tuple(values), Type::Tuple(types)) if values.len() == types.len() => {
                Ok(Self::Tuple(
                    values
                        .iter()
                        .zip(types.iter())
                        .map(|(v, t)| v.normalized(module, t))
                        .collect::<Result<_, _>>()?,
                ))
            }
            _ => Err(format!(
                "value does not match declared type {ty:?}; dimensions must be fixed"
            )),
        }
    }

    fn in_set(&self, module: &Module, set: &ValueSet) -> Result<bool, String> {
        use ValueSet::*;
        if let Some(value) = self.numeric_value() {
            let contains = match set {
                Booleans => value == 0.0 || value == 1.0,
                Integers => value.is_finite() && value.fract() == 0.0,
                PosIntegers => value > 0.0 && value.is_finite() && value.fract() == 0.0,
                NonNegIntegers => value >= 0.0 && value.is_finite() && value.fract() == 0.0,
                Reals => !value.is_nan(),
                PosReals => value > 0.0,
                NonNegReals => value >= 0.0,
                UnitInterval => (0.0..=1.0).contains(&value),
                Interval(lo, hi) => *lo <= value && value <= *hi,
                _ => {
                    return match set {
                        Anything => Ok(true),
                        Deferred | Unknown => {
                            Err("external domain cannot be resolved after substitution".into())
                        }
                        _ => Ok(false),
                    };
                }
            };
            return Ok(contains);
        }
        Ok(match (self, set) {
            (_, Anything) => true,
            (Self::Complex(re, im), Complexes) => re.is_finite() && im.is_finite(),
            (_, Deferred | Unknown) => {
                return Err("external domain cannot be resolved after substitution".into());
            }
            (Self::Array { shape, data }, _) => {
                array_in_set(module, shape, data.view(0, data.len()), set)?
            }
            (Self::Record(fields), RecordSet(sets)) => {
                fields.len() == sets.len()
                    && fields
                        .iter()
                        .zip(sets.iter())
                        .map(|((name, value), (key, set))| {
                            Ok(name == module.resolve(*key) && value.in_set(module, set)?)
                        })
                        .collect::<Result<Vec<_>, String>>()?
                        .into_iter()
                        .all(|v| v)
            }
            (Self::Table(columns), CartPow(row, n)) => {
                if let RecordSet(fields) = row.as_ref() {
                    columns.len() == fields.len()
                        && columns
                            .iter()
                            .zip(fields.iter())
                            .map(|((name, value), (key, set))| {
                                Ok(name == module.resolve(*key)
                                    && value.in_set(module, &CartPow(Box::new(set.clone()), *n))?)
                            })
                            .collect::<Result<Vec<_>, String>>()?
                            .into_iter()
                            .all(|v| v)
                } else {
                    false
                }
            }
            _ => false,
        })
    }

    pub(crate) fn alloc(&self, module: &mut Module) -> Result<NodeId, String> {
        let scalar = match self {
            Self::Bool(value) => Some(Scalar::Bool(*value)),
            Self::Integer(value) => Some(Scalar::Int(*value)),
            Self::Real(value) => Some(Scalar::Real(*value)),
            Self::Real32(value) => Some(Scalar::Real(f64::from(*value))),
            Self::String(value) => Some(Scalar::Str(value.clone().into())),
            _ => None,
        };
        if let Some(scalar) = scalar {
            return Ok(module.alloc(Node::Lit(scalar)));
        }
        let (head, args, named) = match self {
            Self::Complex(re, im) => {
                let args = [re, im]
                    .into_iter()
                    .map(|&value| module.alloc(Node::Lit(Scalar::Real(value))))
                    .collect();
                ("complex", args, vec![])
            }
            Self::Array { shape, data } => {
                let size = shape
                    .iter()
                    .try_fold(1_usize, |n, &d| n.checked_mul(d as usize));
                if shape.contains(&0) {
                    // `set` substitutes a literal, and a literal cannot be empty.
                    return Err(
                        "a constant array with a zero extent has no FlatPPL literal: \
                         `array` sizes must be positive (spec §07)"
                            .into(),
                    );
                }
                if shape.is_empty() || size != Some(data.len()) {
                    return Err("constant array shape must match its nonempty flat data".into());
                }
                // Scalars of one kind become one packed node sharing the
                // buffer, not a node each.
                let packed = match data {
                    ArrayData::Bool(v) => Some(Packed::Bool(v.clone())),
                    ArrayData::Integer(v) => Some(Packed::Int(v.clone())),
                    ArrayData::Real(v) => Some(Packed::Real(v.clone())),
                    ArrayData::Real32(v) => {
                        Some(Packed::Real(v.iter().map(|&x| f64::from(x)).collect()))
                    }
                    ArrayData::Complex(_) | ArrayData::Values(_) => None,
                };
                let values = match packed.filter(|p| p.len() >= Packed::MIN_LEN) {
                    Some(packed) => module.alloc(Node::Packed(packed)),
                    None => {
                        let values = data
                            .iter()
                            .map(|x| x.alloc(module))
                            .collect::<Result<Vec<_>, _>>()?;
                        call(module, "vector", values, vec![])
                    }
                };
                if shape.len() == 1 {
                    return Ok(values);
                }
                let sizes = shape
                    .iter()
                    .map(|&d| module.alloc(Node::Lit(Scalar::Int(d.into()))))
                    .collect();
                let sizes = call(module, "vector", sizes, vec![]);
                let order = (1..=shape.len())
                    .map(|d| module.alloc(Node::Lit(Scalar::Int(d as i64))))
                    .collect();
                let order = call(module, "vector", order, vec![]);
                ("array", vec![values, sizes, order], vec![])
            }
            Self::Record(fields) | Self::Table(fields) => {
                let mut named = Vec::new();
                for (name, value) in fields {
                    let name = module.intern(name);
                    if named.iter().any(|field: &NamedArg| field.name == name) {
                        return Err("constant record fields must be unique".into());
                    }
                    named.push(NamedArg {
                        name,
                        value: value.alloc(module)?,
                        kind: NamedKind::Field,
                    });
                }
                (
                    if matches!(self, Self::Table(_)) {
                        "table"
                    } else {
                        "record"
                    },
                    vec![],
                    named,
                )
            }
            Self::Tuple(items) => (
                "tuple",
                items
                    .iter()
                    .map(|x| x.alloc(module))
                    .collect::<Result<_, _>>()?,
                vec![],
            ),
            Self::Fixed(_) | Self::Reference(_) => {
                return Err("`fixed` markers and references are not values".into());
            }
            _ => unreachable!("scalars handled above"),
        };
        Ok(call(module, head, args, named))
    }

    fn real_value(&self) -> Option<f64> {
        match self {
            Self::Real(value) => Some(*value),
            Self::Real32(value) => Some(f64::from(*value)),
            _ => None,
        }
    }

    fn numeric_value(&self) -> Option<f64> {
        match self {
            Self::Bool(value) => Some(f64::from(*value)),
            Self::Integer(value) => Some(*value as f64),
            _ => self.real_value(),
        }
    }

    /// Read a binding's right-hand side back as a value, structurally.
    ///
    /// Literals, `neg` of a number, `inf`, `pi`, `im`, and the `complex`, `vector`, `array`,
    /// `record`, `table`, `tuple` and `fixed` constructors read back. A
    /// reference to another binding reads as [`Constant::Reference`]. Nothing
    /// is evaluated, so any other expression is refused.
    pub(crate) fn read(module: &Module, node: NodeId) -> Result<Self, String> {
        let name = |s: Symbol| module.resolve(s);
        match module.node(node) {
            Node::Lit(Scalar::Bool(v)) => Ok(Self::Bool(*v)),
            Node::Lit(Scalar::Int(v)) => Ok(Self::Integer(*v)),
            Node::Lit(Scalar::Real(v)) => Ok(Self::Real(*v)),
            Node::Lit(Scalar::Str(v)) => Ok(Self::String(v.to_string())),
            Node::Packed(p) => Ok(Self::Array {
                shape: vec![u32::try_from(p.len()).map_err(|_| "vector is too long")?],
                data: match p {
                    Packed::Bool(v) => ArrayData::Bool(v.clone()),
                    Packed::Int(v) => ArrayData::Integer(v.clone()),
                    Packed::Real(v) => ArrayData::Real(v.clone()),
                },
            }),
            Node::Const(s) if name(*s) == "inf" => Ok(Self::Real(f64::INFINITY)),
            Node::Const(s) if name(*s) == "pi" => Ok(Self::Real(std::f64::consts::PI)),
            Node::Const(s) if name(*s) == "im" => Ok(Self::Complex(0.0, 1.0)),
            Node::Ref(r) if r.ns == RefNs::SelfMod => Ok(Self::Reference(name(r.name).into())),
            Node::Call(c) => {
                let CallHead::Builtin(head) = c.head else {
                    return Err("a call is not a literal".into());
                };
                let args = || {
                    c.args
                        .iter()
                        .map(|&a| Self::read(module, a))
                        .collect::<Result<Vec<_>, _>>()
                };
                let fields = || {
                    c.named
                        .iter()
                        .map(|f| Ok((name(f.name).to_string(), Self::read(module, f.value)?)))
                        .collect::<Result<Vec<_>, String>>()
                };
                match (name(head), c.args.len(), c.named.is_empty()) {
                    ("neg", 1, true) => match Self::read(module, c.args[0])? {
                        Self::Integer(v) => Ok(Self::Integer(-v)),
                        Self::Real(v) => Ok(Self::Real(-v)),
                        Self::Complex(re, im) => Ok(Self::Complex(-re, -im)),
                        _ => Err("`neg` of a non-number is not a literal".into()),
                    },
                    ("complex", 2, true) => {
                        let part = |id| {
                            Self::read(module, id)?.numeric_value().ok_or_else(|| {
                                "complex parts must be literal real numbers".to_string()
                            })
                        };
                        Ok(Self::Complex(part(c.args[0])?, part(c.args[1])?))
                    }
                    ("fixed", 1, true) => Ok(Self::Fixed(Box::new(Self::read(module, c.args[0])?))),
                    ("tuple", _, true) => Ok(Self::Tuple(args()?)),
                    ("record", 0, _) => Ok(Self::Record(fields()?)),
                    ("table", 0, _) => Ok(Self::Table(fields()?)),
                    ("vector", n, true) => Ok(Self::Array {
                        shape: vec![u32::try_from(n).map_err(|_| "vector is too long")?],
                        data: scalars(args()?)?.into(),
                    }),
                    ("array", 3, true) => read_array(module, &c.args),
                    (other, _, _) => Err(format!("`{other}(...)` is not a literal")),
                }
            }
            _ => Err("the expression is not a literal".into()),
        }
    }
}

fn scalars(values: Vec<Constant>) -> Result<Vec<Constant>, String> {
    let kind = std::mem::discriminant;
    match values.first() {
        Some(
            first @ (Constant::Bool(_)
            | Constant::Integer(_)
            | Constant::Real(_)
            | Constant::Complex(_, _)),
        ) if values.iter().all(|v| kind(v) == kind(first)) => Ok(values),
        None => Ok(values),
        _ => Err("array elements must be literal numbers or booleans of one kind".into()),
    }
}

fn read_array(module: &Module, args: &[NodeId]) -> Result<Constant, String> {
    let integers = |node| match Constant::read(module, node)? {
        Constant::Array { data, .. } => data
            .iter()
            .map(|v| match v {
                Constant::Integer(v) => u32::try_from(v).map_err(|_| "negative array extent"),
                _ => Err("array extents must be integers"),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(String::from),
        _ => Err("array extents must be a vector".to_string()),
    };
    let Constant::Array { data, .. } = Constant::read(module, args[0])? else {
        return Err("array data must be a vector".into());
    };
    let shape = integers(args[1])?;
    let order = integers(args[2])?;
    if !order.iter().copied().eq(1..=shape.len() as u32) {
        return Err("only arrays in the default axis order read back".into());
    }
    if shape.iter().map(|&d| d as usize).product::<usize>() != data.len() {
        return Err("array data does not match its shape".into());
    }
    Ok(Constant::Array { shape, data })
}

fn normalize_fields(
    module: &Module,
    fields: &[(String, Constant)],
    expected: &[(Symbol, Type)],
) -> Result<Vec<(String, Constant)>, String> {
    if fields.len() != expected.len() {
        return Err("field names must match the external declaration".into());
    }
    expected
        .iter()
        .map(|(name, ty)| {
            let name = module.resolve(*name);
            let (_, value) = fields
                .iter()
                .find(|(key, _)| key == name)
                .ok_or_else(|| format!("missing field `{name}`"))?;
            Ok((name.into(), value.normalized(module, ty)?))
        })
        .collect()
}

fn array_in_set(
    module: &Module,
    shape: &[u32],
    data: ArrayView<'_>,
    set: &ValueSet,
) -> Result<bool, String> {
    use ValueSet::*;
    if shape.is_empty() {
        return data
            .get(0)
            .ok_or("an array entry is missing")?
            .in_set(module, set);
    }
    match set {
        CartPow(elem, Dim::Static(n)) if *n == shape[0] => {
            if *n == 0 {
                return Ok(data.len == 0);
            }
            let width = data.len / *n as usize;
            // Rows as `chunks(width)` cuts them, the last one possibly short.
            let mut start = 0;
            while start < data.len {
                let row = data
                    .sub(start, width.min(data.len - start))
                    .expect("row within the data");
                if !array_in_set(module, &shape[1..], row, elem)? {
                    return Ok(false);
                }
                start += width.max(1);
            }
            Ok(true)
        }
        StdSimplex(Dim::Static(n)) if shape == [*n] => {
            for value in data.iter() {
                if !value.real_value().is_some_and(|v| v >= 0.0) {
                    return Ok(false);
                }
            }
            Ok(sum_matches(data, 1.0))
        }
        Ordered(Dim::Static(n)) if shape == [*n] => {
            let mut previous = f64::NEG_INFINITY;
            for (i, value) in data.iter().enumerate() {
                let Some(value) = value.real_value() else {
                    return Ok(false);
                };
                if value.is_nan() || (i > 0 && value <= previous) {
                    return Ok(false);
                }
                previous = value;
            }
            Ok(true)
        }
        ZeroSum(Dim::Static(n)) if shape == [*n] => Ok(sum_matches(data, 0.0)),
        CorrCholesky(Dim::Static(n)) if shape == [*n, *n] => {
            Ok(corr_cholesky_matches(data, *n as usize))
        }
        CartProd(parts) if shape.len() == 1 => {
            let mut offset = 0;
            for part in parts {
                let width = product_width(part).ok_or("unresolved Cartesian product domain")?;
                let Some(values) = data.sub(offset, width) else {
                    return Ok(false);
                };
                let dims = match part {
                    CartPow(_, _) | CartProd(_) | StdSimplex(_) | Ordered(_) | ZeroSum(_) => {
                        vec![width as u32]
                    }
                    _ => vec![],
                };
                if !array_in_set(module, &dims, values, part)? {
                    return Ok(false);
                }
                offset += width;
            }
            Ok(offset == data.len)
        }
        Deferred
        | Unknown
        | CartPow(_, Dim::Dynamic)
        | StdSimplex(Dim::Dynamic)
        | Ordered(Dim::Dynamic)
        | ZeroSum(Dim::Dynamic)
        | CorrCholesky(Dim::Dynamic) => {
            Err("external domain cannot be resolved after substitution".into())
        }
        _ => Ok(false),
    }
}

/// Fixed-data acceptance only: |sum(x) - target| <= 8 n u sum(|x|).
/// The factor eight allows a short producing chain per entry. Scaling avoids
/// overflow without changing the stored values; mixed reals promote to f64.
fn sum_matches(data: ArrayView<'_>, target: f64) -> bool {
    let mut scale = 0.0_f64;
    let mut epsilon = f64::from(f32::EPSILON);
    for value in data.iter() {
        let Some(number) = value.real_value().filter(|v| v.is_finite()) else {
            return false;
        };
        scale = scale.max(number.abs());
        if matches!(value, Constant::Real(_)) {
            epsilon = f64::EPSILON;
        }
    }
    if scale == 0.0 {
        return target == 0.0;
    }
    let (mut sum, mut abs_sum) = (0.0, 0.0);
    for value in data.iter() {
        let number = value.real_value().unwrap() / scale;
        sum += number;
        abs_sum += number.abs();
    }
    (sum - target / scale).abs() <= 8.0 * data.len as f64 * (epsilon / 2.0) * abs_sum
}

/// Fixed-data acceptance for a row-major n×n `corrcholesky(n)` value: finite
/// entries, an exactly zero upper triangle, a positive diagonal and unit row
/// norms, |sum_j L_ij^2 - 1| <= 8 n u, with the unit roundoff `u` of
/// [`sum_matches`] (f32 unless an entry is a real). The entries are at most one
/// in magnitude, so the squares need no scaling.
fn corr_cholesky_matches(data: ArrayView<'_>, n: usize) -> bool {
    if n == 0 {
        return data.len == 0;
    }
    let mut epsilon = f64::from(f32::EPSILON);
    let mut values = Vec::with_capacity(data.len);
    for value in data.iter() {
        let Some(number) = value.real_value().filter(|v| v.is_finite()) else {
            return false;
        };
        if matches!(value, Constant::Real(_)) {
            epsilon = f64::EPSILON;
        }
        values.push(number);
    }
    let tolerance = 8.0 * n as f64 * (epsilon / 2.0);
    values.chunks(n).enumerate().all(|(i, row)| {
        let norm: f64 = row[..=i].iter().map(|v| v * v).sum();
        row[i] > 0.0 && row[i + 1..].iter().all(|&v| v == 0.0) && (norm - 1.0).abs() <= tolerance
    })
}

fn product_width(set: &ValueSet) -> Option<usize> {
    match set {
        ValueSet::CartPow(_, Dim::Static(n))
        | ValueSet::StdSimplex(Dim::Static(n))
        | ValueSet::Ordered(Dim::Static(n))
        | ValueSet::ZeroSum(Dim::Static(n)) => Some(*n as usize),
        ValueSet::CartProd(parts) => parts
            .iter()
            .try_fold(0_usize, |n, set| n.checked_add(product_width(set)?)),
        ValueSet::Reals
        | ValueSet::PosReals
        | ValueSet::NonNegReals
        | ValueSet::UnitInterval
        | ValueSet::Integers
        | ValueSet::PosIntegers
        | ValueSet::NonNegIntegers
        | ValueSet::Booleans
        | ValueSet::Interval(_, _) => Some(1),
        _ => None,
    }
}

pub(crate) fn call(
    module: &mut Module,
    head: &str,
    args: Vec<NodeId>,
    named: Vec<NamedArg>,
) -> NodeId {
    let head = CallHead::Builtin(module.intern(head));
    module.alloc(Node::Call(Call {
        head,
        args: args.into(),
        named: named.into(),
        inputs: None,
    }))
}
