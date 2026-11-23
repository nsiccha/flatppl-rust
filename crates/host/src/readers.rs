//! Decode a `load_data` source by its extension (spec §07 `load_data`): JSON, CSV
//! and WSV, Arrow IPC, and Safetensors, each through its one reader in
//! `flatppl-data`. Decoding infers no FlatPPL shape. The declared value set
//! selects and checks the value ([`crate::data`]).

use flatppl_data::safetensors::{self, TensorValues};
use flatppl_data::{ColumnValues, Table};
use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};

/// A decoded source, before the declared value set selects from it.
#[derive(Clone, Debug)]
pub(crate) enum Datum {
    /// A missing cell: an empty delimited field, an Arrow null, a JSON null.
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    /// A JSON number written with a fraction or an exponent. JSON has one
    /// number type, so its value, not its spelling, decides its set.
    Number(f64),
    /// A binary32 (or narrower) value, kept at its source precision.
    Real32(f32),
    /// A delimited cell, parsed only once its scalar set is known.
    Text(String),
    List(Vec<Datum>),
    /// A list of scalars of one kind, unboxed: the [`Datum::List`] of
    /// those scalars at eight bytes or less per entry.
    Leaves(Leaves),
    /// A JSON object or a Safetensors record, fields in source order.
    Map(Vec<(String, Datum)>),
    /// Named columns of a delimited or Arrow table.
    Columns(Vec<(String, Datum)>),
    /// A Safetensors tensor: row-major leaves of a fixed shape.
    Tensor {
        shape: Vec<usize>,
        data: Leaves,
    },
    /// A column no value set can select, with the reason. It refuses only when
    /// selected.
    Unreadable(String),
}

/// Scalars of one kind; entry `i` reads as the scalar [`Datum`] of that kind.
#[derive(Clone, Debug)]
pub(crate) enum Leaves {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    Real(Vec<f64>),
    /// JSON numbers. A JSON list mixing integers and numbers holds its
    /// integers here as the equal numbers when every one is exact in binary64.
    Number(Vec<f64>),
    Real32(Vec<f32>),
}

impl Leaves {
    pub(crate) fn len(&self) -> usize {
        match self {
            Leaves::Bool(v) => v.len(),
            Leaves::Int(v) => v.len(),
            Leaves::Real(v) | Leaves::Number(v) => v.len(),
            Leaves::Real32(v) => v.len(),
        }
    }

    pub(crate) fn get(&self, i: usize) -> Option<Datum> {
        Some(match self {
            Leaves::Bool(v) => Datum::Bool(*v.get(i)?),
            Leaves::Int(v) => Datum::Int(*v.get(i)?),
            Leaves::Real(v) => Datum::Real(*v.get(i)?),
            Leaves::Number(v) => Datum::Number(*v.get(i)?),
            Leaves::Real32(v) => Datum::Real32(*v.get(i)?),
        })
    }
}

/// Integers up to this magnitude are exact in binary64.
const EXACT_INT: i64 = 1 << 53;

/// A list being decoded: unboxed while its entries are scalars of one kind.
enum ListBuilder {
    Leaves(Leaves),
    General(Vec<Datum>),
}

impl ListBuilder {
    fn push(&mut self, item: Datum) {
        let leaves = match self {
            ListBuilder::General(items) => {
                items.push(item);
                return;
            }
            ListBuilder::Leaves(leaves) => leaves,
        };
        match (leaves, item) {
            (Leaves::Bool(v), Datum::Bool(b)) => v.push(b),
            (Leaves::Int(v), Datum::Int(i)) => v.push(i),
            (Leaves::Number(v), Datum::Number(x)) => v.push(x),
            (Leaves::Number(v), Datum::Int(i)) if i.abs() <= EXACT_INT => v.push(i as f64),
            (Leaves::Int(v), Datum::Number(x)) if v.iter().all(|i| i.abs() <= EXACT_INT) => {
                let mut numbers: Vec<f64> = v.iter().map(|&i| i as f64).collect();
                numbers.push(x);
                *self = ListBuilder::Leaves(Leaves::Number(numbers));
            }
            (leaves, item) => {
                let mut items: Vec<Datum> = (0..leaves.len())
                    .map(|i| leaves.get(i).expect("index below len"))
                    .collect();
                items.push(item);
                *self = ListBuilder::General(items);
            }
        }
    }

    fn finish(self) -> Datum {
        match self {
            ListBuilder::Leaves(leaves) => Datum::Leaves(leaves),
            ListBuilder::General(items) => Datum::List(items),
        }
    }
}

/// Decodes JSON straight into a [`Datum`], with no intermediate document tree.
struct JsonDatum;

impl<'de> Visitor<'de> for JsonDatum {
    type Value = Datum;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<Datum, E> {
        Ok(Datum::Null)
    }

    fn visit_bool<E>(self, b: bool) -> Result<Datum, E> {
        Ok(Datum::Bool(b))
    }

    fn visit_i64<E>(self, i: i64) -> Result<Datum, E> {
        Ok(Datum::Int(i))
    }

    fn visit_u64<E>(self, u: u64) -> Result<Datum, E> {
        Ok(match i64::try_from(u) {
            Ok(i) => Datum::Int(i),
            Err(_) => Datum::Unreadable(format!("integer {u} exceeds int64")),
        })
    }

    fn visit_f64<E>(self, x: f64) -> Result<Datum, E> {
        Ok(Datum::Number(x))
    }

    // JSON types its values, and a string is not a number.
    fn visit_str<E>(self, s: &str) -> Result<Datum, E> {
        Ok(Datum::Unreadable(format!(
            "JSON string {s:?} is not a number or a boolean"
        )))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Datum, A::Error> {
        let mut list = None::<ListBuilder>;
        while let Some(item) = seq.next_element_seed(JsonSeed)? {
            match &mut list {
                Some(list) => list.push(item),
                None => {
                    list = Some(match item {
                        Datum::Bool(b) => ListBuilder::Leaves(Leaves::Bool(vec![b])),
                        Datum::Int(i) => ListBuilder::Leaves(Leaves::Int(vec![i])),
                        Datum::Number(x) => ListBuilder::Leaves(Leaves::Number(vec![x])),
                        item => ListBuilder::General(vec![item]),
                    })
                }
            }
        }
        Ok(list.map_or(Datum::List(Vec::new()), ListBuilder::finish))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Datum, A::Error> {
        // Fields in key order, a repeated key keeping its last value, as a
        // JSON object document reads.
        let mut fields: Vec<(String, Datum)> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = map.next_value_seed(JsonSeed)?;
            match fields.binary_search_by(|(k, _)| k.as_str().cmp(&key)) {
                Ok(at) => fields[at].1 = value,
                Err(at) => fields.insert(at, (key, value)),
            }
        }
        Ok(Datum::Map(fields))
    }
}

struct JsonSeed;

impl<'de> serde::de::DeserializeSeed<'de> for JsonSeed {
    type Value = Datum;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Datum, D::Error> {
        deserializer.deserialize_any(JsonDatum)
    }
}

impl Datum {
    /// The named entry of a record-like datum.
    pub(crate) fn field(&self, name: &str) -> Option<&Datum> {
        match self {
            Datum::Map(fields) | Datum::Columns(fields) => {
                fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
            }
            _ => None,
        }
    }
}

/// The formats §07 requires, by extension.
pub(crate) enum Format {
    Json,
    Csv,
    Wsv,
    Arrow,
    Safetensors,
}

/// The format of `source` from its extension, ignoring a URL query or fragment.
pub(crate) fn format(source: &str) -> Result<Format, String> {
    let path = source.split(['?', '#']).next().unwrap_or(source);
    let name = path.rsplit('/').next().unwrap_or(path);
    let extension = name
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "json" => Ok(Format::Json),
        "csv" => Ok(Format::Csv),
        "wsv" => Ok(Format::Wsv),
        "arrow" | "arrows" => Ok(Format::Arrow),
        "safetensors" => Ok(Format::Safetensors),
        _ => Err(format!(
            "unsupported file extension {}; §07 `load_data` reads `.json`, `.csv`, `.wsv`, \
             `.arrow`, `.arrows` and `.safetensors`",
            if extension.is_empty() {
                "(none)".to_owned()
            } else {
                format!("`.{extension}`")
            }
        )),
    }
}

/// Decode `bytes` in `format`.
pub(crate) fn decode(format: Format, bytes: &[u8]) -> Result<Datum, String> {
    match format {
        Format::Json => {
            let mut deserializer = serde_json::Deserializer::from_slice(bytes);
            let datum = deserializer
                .deserialize_any(JsonDatum)
                .and_then(|datum| deserializer.end().map(|()| datum))
                .map_err(|e| format!("JSON: {e}"))?;
            Ok(datum)
        }
        Format::Csv => flatppl_data::read_csv(bytes)
            .map(text_columns)
            .map_err(|e| e.0),
        Format::Wsv => flatppl_data::read_wsv(bytes)
            .map(text_columns)
            .map_err(|e| e.0),
        Format::Arrow => Table::from_arrow_ipc_values(bytes)
            .map(arrow_columns)
            .map_err(|e| e.0),
        Format::Safetensors => safetensors::read(bytes)
            .map(safetensors_entry)
            .map_err(|e| e.0),
    }
}

/// Columns of text cells from a delimited source. An empty cell is missing.
fn text_columns(table: flatppl_data::TextTable) -> Datum {
    Datum::Columns(
        table
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let cells = table
                    .column(i)
                    .map(|cell| {
                        if cell.is_empty() {
                            Datum::Null
                        } else {
                            Datum::Text(cell.to_owned())
                        }
                    })
                    .collect();
                (name.clone(), Datum::List(cells))
            })
            .collect(),
    )
}

fn arrow_columns(table: Table) -> Datum {
    let mut columns: Vec<(String, Datum)> = table
        .columns
        .into_iter()
        .map(|column| {
            let flatppl_data::Column {
                name,
                values,
                valid,
            } = column;
            let valid = |i: usize| valid.as_ref().is_none_or(|v| v[i]);
            let cells = match values {
                ColumnValues::Real32(v) => Leaves::Real32(v),
                ColumnValues::Real64(v) => Leaves::Real(v),
                ColumnValues::Int(v) => Leaves::Int(v),
                ColumnValues::Bool(v) => Leaves::Bool(v),
                // Arrow is typed, and a string is not a number.
                ColumnValues::Factor { .. } => {
                    let reason = format!(
                        "column `{name}` holds Arrow strings, and `load_data` reads numbers and \
                         booleans"
                    );
                    return (name, Datum::Unreadable(reason));
                }
            };
            // A null cell is a missing value, which only a boxed list holds.
            if (0..cells.len()).all(valid) {
                return (name, Datum::Leaves(cells));
            }
            let cells = (0..cells.len())
                .map(|i| {
                    if valid(i) {
                        cells.get(i).expect("index below len")
                    } else {
                        Datum::Null
                    }
                })
                .collect();
            (name, Datum::List(cells))
        })
        .collect();
    columns.extend(
        table
            .unreadable
            .into_iter()
            .map(|(name, reason)| (name, Datum::Unreadable(reason))),
    );
    Datum::Columns(columns)
}

/// A Safetensors record tree, each tensor's leaves at their source kind.
fn safetensors_entry(entry: safetensors::Entry) -> Datum {
    match entry {
        safetensors::Entry::Record(fields) => Datum::Map(
            fields
                .into_iter()
                .map(|(name, entry)| (name, safetensors_entry(entry)))
                .collect(),
        ),
        safetensors::Entry::Tensor(tensor) => {
            let data = match tensor.values {
                TensorValues::Real32(v) => Leaves::Real32(v),
                TensorValues::Real64(v) => Leaves::Real(v),
                TensorValues::Int(v) => Leaves::Int(v),
                TensorValues::Bool(v) => Leaves::Bool(v),
            };
            Datum::Tensor {
                shape: tensor.shape,
                data,
            }
        }
    }
}
