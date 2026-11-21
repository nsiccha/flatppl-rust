//! Guard the built-in roster (`flatppl_infer::builtins::BUILTINS`, re-exported
//! by `flatppl_lint::builtins`) against drift from the authoritative
//! `flatppl-grammars/keyword-lists.json`. Skipped (passes) when the sibling repo
//! is not checked out, so CI without it still builds.
//!
//! The roster also feeds spec-§04 name resolution now
//! (`flatppl_infer::builtins::is_base_name`), so a silent skip here is worse than
//! it was: it would let the list drift under the resolver. Hence
//! `keyword_lists_path` searches the ancestors instead of counting `..`
//! segments — a fixed `../../../` resolves only in the primary checkout and
//! missed the file (skipping the test) from a git worktree, whose crates sit two
//! levels deeper.
//!
//! **This guard still does not run in CI, and the ancestor search does not change
//! that.** CI checks out `flatppl-rust` (and `flatppl-js`) only, so no ancestor
//! holds `flatppl-grammars` and the test returns early — and because it PASSES
//! when it skips, the `eprintln` is captured and invisible. So drift between the
//! roster and `keyword-lists.json` is caught only by a local run in a full
//! workspace checkout. Closing that needs a cross-repo CI gate in
//! `flatppl-grammars`; it is a follow-up there, not something this test can fix.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The nearest ancestor of this crate holding `flatppl-grammars/keyword-lists.json`.
fn keyword_lists_path() -> Option<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|d| d.join("flatppl-grammars/keyword-lists.json"))
        .find(|p| p.exists())
}

/// Names the dogfood fork adds before `flatppl-grammars` has them, so they are
/// in `BUILTINS` but not yet in `keyword-lists.json`. `corrcholesky` is the
/// support set of `LKJCholesky` (fork-only until flatppl-design adopts it).
/// Drop a name here once the keyword list carries it.
const FORK_ONLY: &[&str] = &["corrcholesky"];

#[test]
fn builtins_match_keyword_lists() {
    let Some(path) = keyword_lists_path() else {
        eprintln!("skipping: no flatppl-grammars/keyword-lists.json in any ancestor");
        return;
    };
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let expected: BTreeSet<String> = json["categories"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c["words"].as_array().unwrap())
        .map(|w| w.as_str().unwrap().to_string())
        .collect();
    for name in FORK_ONLY {
        assert!(
            !expected.contains(*name),
            "keyword-lists.json now has `{name}`: remove it from FORK_ONLY"
        );
    }
    let actual: BTreeSet<String> = flatppl_lint::test_builtins()
        .iter()
        .filter(|s| !FORK_ONLY.contains(*s))
        .map(|s| s.to_string())
        .collect();
    assert_eq!(actual, expected, "BUILTINS drifted from keyword-lists.json");
}

/// Every fork-only name is a real roster entry, so the divergence list above
/// cannot hide a stale name.
#[test]
fn fork_only_names_are_builtins() {
    for name in FORK_ONLY {
        assert!(
            flatppl_lint::test_builtins().contains(name),
            "`{name}` is listed as fork-only but is not in BUILTINS"
        );
    }
}

/// `shadows-builtin` looks up names with `binary_search`, which requires the
/// `BUILTINS` slice to be sorted and duplicate-free. This pins that invariant
/// independently of the sibling repo (always runs).
#[test]
fn builtins_sorted_and_unique() {
    let b = flatppl_lint::test_builtins();
    assert!(
        b.windows(2).all(|w| w[0] < w[1]),
        "BUILTINS must be strictly sorted for binary_search"
    );
}
