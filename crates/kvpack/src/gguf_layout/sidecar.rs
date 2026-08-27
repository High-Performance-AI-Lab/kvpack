use super::*;

/// Derive a v2 layout from a JSON sidecar descriptor on disk. The sidecar is
/// the escape hatch for geometry GGUF cannot express (per-class KV-head
/// counts, explicit per-layer class lists).
pub fn derive_layout_from_sidecar(path: &Path) -> Result<OwnedLayoutV2, StoreError> {
    let json = std::fs::read_to_string(path).map_err(io_error("read layout sidecar"))?;
    parse_layout_sidecar(&json)
}

/// Parse and validate a sidecar descriptor:
///
/// ```json
/// {
///   "name": "gemma4-31b",
///   "num_layers": 60,
///   "classes": [
///     {"class": "gqa-windowed", "from": 0, "until": 60, "step": 1,
///      "except": [5, 11], "kv_heads": 16, "head_dim": 256, "window_tokens": 1024,
///      "rope": {"freq_base": 10000.0, "dimension_count": 256,
///               "scaling": "none", "convention": "neox"}},
///     {"class": "gqa-full", "from": 5, "until": 60, "step": 6,
///      "except": [], "kv_heads": 4, "head_dim": 512, "window_tokens": 0,
///      "rope": {"freq_base": 1000000.0, "dimension_count": 512,
///               "scaling": "none", "convention": "neox"}}
///   ]
/// }
/// ```
///
/// Validation is fail-closed: the classes must partition `0..num_layers`
/// without overlap, `window_tokens > 0` is only legal on windowed
/// (non-`gqa-full`) classes, and every class carries an explicit `rope`
/// object — `freq_base` (finite, > 1), `dimension_count` (even, in
/// `2..=head_dim`), `scaling` (canonical label: `none` or
/// `{linear|ntk|yarn}:{factor}`), and `convention` (`neox` or
/// `interleaved`). The rope fields bind into the authenticated identity
/// exactly like the GGUF-derived ones (docs/KV_ALGEBRA_2026-08-09.md).
pub fn parse_layout_sidecar(json: &str) -> Result<OwnedLayoutV2, StoreError> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|_| StoreError::Expectation("prefill layout sidecar is not valid json"))?;
    let name = value
        .get("name")
        .and_then(serde_json::Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or(StoreError::Expectation(
            "prefill layout sidecar name is missing or empty",
        ))?
        .to_string();
    let num_layers = sidecar_u64(&value, "num_layers")?;
    if num_layers == 0 || num_layers > MAX_LAYOUT_LAYERS {
        return Err(StoreError::Expectation(
            "prefill layout sidecar num_layers is outside the v2 bounds",
        ));
    }
    let num_layers = num_layers as u32;
    let classes_value = value
        .get("classes")
        .and_then(serde_json::Value::as_array)
        .filter(|classes| !classes.is_empty())
        .ok_or(StoreError::Expectation(
            "prefill layout sidecar classes are missing or empty",
        ))?;
    let mut classes = Vec::with_capacity(classes_value.len());
    for class_value in classes_value {
        let class = class_value
            .get("class")
            .and_then(serde_json::Value::as_str)
            .filter(|class| !class.is_empty())
            .ok_or(StoreError::Expectation(
                "prefill layout sidecar class name is missing or empty",
            ))?
            .to_string();
        let from = sidecar_u32(class_value, "from")?;
        let until = sidecar_u32(class_value, "until")?;
        let step = sidecar_u32(class_value, "step")?;
        let kv_heads = sidecar_u32(class_value, "kv_heads")?;
        let head_dim = sidecar_u32(class_value, "head_dim")?;
        let window_tokens = sidecar_u32(class_value, "window_tokens")?;
        let except = class_value
            .get("except")
            .and_then(serde_json::Value::as_array)
            .ok_or(StoreError::Expectation(
                "prefill layout sidecar class except list is missing",
            ))?
            .iter()
            .map(|layer| {
                let layer = layer.as_u64().ok_or(StoreError::Expectation(
                    "prefill layout sidecar class except entry is not an integer",
                ))?;
                u32::try_from(layer).map_err(|_| {
                    StoreError::Expectation("prefill layout sidecar numeric field exceeds u32::MAX")
                })
            })
            .collect::<Result<Vec<u32>, StoreError>>()?;
        {
            let mut distinct = BTreeSet::new();
            if except.iter().any(|layer| !distinct.insert(*layer)) {
                return Err(StoreError::Expectation(
                    "prefill layout sidecar class except list contains a duplicate entry",
                ));
            }
        }
        if from >= until || until > num_layers || step == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(StoreError::Expectation(
                "prefill layout sidecar class bounds are outside the v2 bounds",
            ));
        }
        if except.iter().any(|layer| *layer < from || *layer >= until) {
            return Err(StoreError::Expectation(
                "prefill layout sidecar class except list is outside the class span",
            ));
        }
        // mla-latent classes must declare the record split explicitly and
        // carry the packed record as one vector: latent_dim + rope_dim must
        // equal head_dim, and kv_heads must be 1. The fields are only legal
        // on mla-latent classes — anywhere else they are an unknown-geometry
        // smell and refused.
        let latent_dim = class_value.get("latent_dim");
        let rope_dim = class_value.get("rope_dim");
        if class == crate::mla::MLA_LATENT_LAYOUT_CLASS {
            let latent_dim = latent_dim
                .and_then(serde_json::Value::as_u64)
                .filter(|value| *value > 0)
                .ok_or(StoreError::Expectation(
                    "prefill layout sidecar mla-latent class latent_dim is missing or not a positive integer",
                ))?;
            let rope_dim = rope_dim
                .and_then(serde_json::Value::as_u64)
                .filter(|value| *value > 0)
                .ok_or(StoreError::Expectation(
                    "prefill layout sidecar mla-latent class rope_dim is missing or not a positive integer",
                ))?;
            let record_elements = latent_dim.checked_add(rope_dim).ok_or(StoreError::Expectation(
                "prefill layout sidecar mla-latent record must be one vector of latent_dim + rope_dim elements",
            ))?;
            if kv_heads != 1 || record_elements != u64::from(head_dim) {
                return Err(StoreError::Expectation(
                    "prefill layout sidecar mla-latent record must be one vector of latent_dim + rope_dim elements",
                ));
            }
        } else if latent_dim.is_some() || rope_dim.is_some() {
            return Err(StoreError::Expectation(
                "prefill layout sidecar latent_dim/rope_dim are only legal on mla-latent classes",
            ));
        }
        // Full-coverage classes (gqa-full, mla-latent) must have
        // window_tokens == 0; windowed classes must have window_tokens > 0.
        let windowed_class = class != "gqa-full" && class != crate::mla::MLA_LATENT_LAYOUT_CLASS;
        if (window_tokens > 0) != windowed_class {
            return Err(StoreError::Expectation(
                "prefill layout sidecar window_tokens are only legal on windowed classes",
            ));
        }
        // RoPE configuration: required on every class, fail-closed. The
        // sidecar arms geometry GGUF cannot express, so it states the rotary
        // parameters explicitly; they bind into the authenticated identity
        // exactly like the GGUF-derived fields.
        let rope = class_value.get("rope").ok_or(StoreError::Expectation(
            "prefill layout sidecar class rope is missing",
        ))?;
        let rope_freq_base = rope
            .get("freq_base")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value > 1.0)
            .ok_or(StoreError::Expectation(
                "prefill layout sidecar class rope freq_base is missing or outside the supported bounds",
            ))?;
        let rope_dimension_count = sidecar_u32(rope, "dimension_count")?;
        if rope_dimension_count < 2
            || rope_dimension_count % 2 != 0
            || rope_dimension_count > head_dim
        {
            return Err(StoreError::Expectation(
                "prefill layout sidecar class rope dimension_count is outside the v2 bounds",
            ));
        }
        let rope_scaling = rope
            .get("scaling")
            .and_then(serde_json::Value::as_str)
            .filter(|label| is_canonical_rope_scaling(label))
            .ok_or(StoreError::Expectation(
                "prefill layout sidecar class rope scaling is missing or not a canonical label",
            ))?
            .to_string();
        let rope_convention = rope
            .get("convention")
            .and_then(serde_json::Value::as_str)
            .and_then(RopeConvention::from_label)
            .ok_or(StoreError::Expectation(
                "prefill layout sidecar class rope convention is missing or outside the closed set",
            ))?;
        classes.push(OwnedLayoutClassV2 {
            class,
            from,
            until,
            step,
            except,
            kv_heads,
            head_dim,
            window_tokens,
            rope_freq_base_bits: rope_freq_base.to_bits(),
            rope_dimension_count,
            rope_scaling,
            rope_convention,
        });
    }
    // The classes must partition 0..num_layers exactly: every layer covered
    // once, no overlaps, no holes.
    let mut covered = BTreeSet::new();
    for class in &classes {
        for layer in class.layers() {
            if !covered.insert(layer) {
                return Err(StoreError::Expectation(
                    "prefill layout sidecar classes overlap",
                ));
            }
        }
    }
    if covered.len() != num_layers as usize
        || covered.iter().next() != Some(&0)
        || covered.iter().next_back() != Some(&(num_layers - 1))
    {
        return Err(StoreError::Expectation(
            "prefill layout sidecar classes do not partition the layer range",
        ));
    }
    Ok(OwnedLayoutV2 {
        name,
        num_layers,
        classes,
    })
}

fn sidecar_u64(value: &serde_json::Value, field: &str) -> Result<u64, StoreError> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or(StoreError::Expectation(
            "prefill layout sidecar field is missing or not an unsigned integer",
        ))
}

/// Read a numeric sidecar field, refusing values that would truncate in the
/// u32 layout table — no `as u32` cast happens before this range check.
fn sidecar_u32(value: &serde_json::Value, field: &str) -> Result<u32, StoreError> {
    u32::try_from(sidecar_u64(value, field)?).map_err(|_| {
        StoreError::Expectation("prefill layout sidecar numeric field exceeds u32::MAX")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMMA4_31B_SIDECAR: &str = r#"{
        "name": "gemma4-31b",
        "num_layers": 60,
        "classes": [
            {"class": "gqa-windowed", "from": 0, "until": 60, "step": 1,
             "except": [5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
             "kv_heads": 16, "head_dim": 256, "window_tokens": 1024,
             "rope": {"freq_base": 10000.0, "dimension_count": 256,
                      "scaling": "none", "convention": "neox"}},
            {"class": "gqa-full", "from": 5, "until": 60, "step": 6,
             "except": [], "kv_heads": 4, "head_dim": 512, "window_tokens": 0,
             "rope": {"freq_base": 1000000.0, "dimension_count": 512,
                      "scaling": "none", "convention": "neox"}}
        ]
    }"#;

    #[test]
    fn sidecar_reproduces_the_gemma4_31b_registry_entry() {
        let derived = parse_layout_sidecar(GEMMA4_31B_SIDECAR).unwrap();
        assert_eq!(derived.name, "gemma4-31b");
        assert_layout_matches_registry(&derived, "gemma4-31b");
    }

    #[test]
    fn sidecar_round_trip_through_a_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("gemma4-31b.layout.json");
        std::fs::write(&path, GEMMA4_31B_SIDECAR).unwrap();
        let derived = derive_layout_from_sidecar(&path).unwrap();
        assert_layout_matches_registry(&derived, "gemma4-31b");
    }

    #[test]
    fn refuses_numeric_fields_above_u32_max() {
        // 2^32 must not silently truncate to zero in the u32 layout table.
        let json = r#"{
            "name": "huge-heads",
            "num_layers": 2,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 2, "step": 1,
                 "except": [], "kv_heads": 4294967296, "head_dim": 64, "window_tokens": 0}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar numeric field exceeds u32::MAX"
        );

        // Same guard on except entries.
        let json = r#"{
            "name": "huge-except",
            "num_layers": 2,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 2, "step": 1,
                 "except": [4294967296], "kv_heads": 4, "head_dim": 64, "window_tokens": 0}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar numeric field exceeds u32::MAX"
        );
    }

    #[test]
    fn refuses_num_layers_above_the_layout_cap() {
        // Four billion layers would explode the per-layer expansion long
        // before any geometry check; the derivation cap refuses it.
        let json = r#"{
            "name": "four-billion-layers",
            "num_layers": 4000000000,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 1, "step": 1,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar num_layers is outside the v2 bounds"
        );
    }

    #[test]
    fn refuses_duplicate_except_entries() {
        let json = r#"{
            "name": "dup-except",
            "num_layers": 4,
            "classes": [
                {"class": "gqa-windowed", "from": 0, "until": 4, "step": 1,
                 "except": [1, 1, 3], "kv_heads": 4, "head_dim": 64, "window_tokens": 128},
                {"class": "gqa-full", "from": 1, "until": 4, "step": 2,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar class except list contains a duplicate entry"
        );
    }

    #[test]
    fn refuses_malformed_sidecar_json() {
        let error = parse_layout_sidecar("{ not json").unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar is not valid json"
        );
    }

    #[test]
    fn refuses_overlapping_sidecar_classes() {
        let json = r#"{
            "name": "overlap",
            "num_layers": 4,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 3, "step": 1,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0,
                 "rope": {"freq_base": 500000.0, "dimension_count": 64,
                          "scaling": "none", "convention": "neox"}},
                {"class": "gqa-full", "from": 2, "until": 4, "step": 1,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0,
                 "rope": {"freq_base": 500000.0, "dimension_count": 64,
                          "scaling": "none", "convention": "neox"}}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(error.to_string(), "prefill layout sidecar classes overlap");
    }

    #[test]
    fn refuses_sidecar_classes_that_do_not_cover_every_layer() {
        let json = r#"{
            "name": "hole",
            "num_layers": 4,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 3, "step": 1,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0,
                 "rope": {"freq_base": 500000.0, "dimension_count": 64,
                          "scaling": "none", "convention": "neox"}}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar classes do not partition the layer range"
        );
    }

    #[test]
    fn refuses_window_tokens_on_a_full_attention_class() {
        let json = r#"{
            "name": "bad-window",
            "num_layers": 2,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 2, "step": 1,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 1024}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar window_tokens are only legal on windowed classes"
        );
    }

    const MLA_SIDECAR: &str = r#"{
        "name": "deepseek-mla-fixture",
        "num_layers": 2,
        "classes": [
            {"class": "mla-latent", "from": 0, "until": 2, "step": 1,
             "except": [], "kv_heads": 1, "head_dim": 20, "window_tokens": 0,
             "latent_dim": 16, "rope_dim": 4,
             "rope": {"freq_base": 10000.0, "dimension_count": 4,
                      "scaling": "none", "convention": "neox"}}
        ]
    }"#;

    #[test]
    fn accepts_an_mla_latent_class_with_a_valid_record_split() {
        let derived = parse_layout_sidecar(MLA_SIDECAR).unwrap();
        assert_eq!(derived.name, "deepseek-mla-fixture");
        assert_eq!(derived.num_layers, 2);
        let class = &derived.classes[0];
        assert_eq!(class.class, "mla-latent");
        assert_eq!(class.kv_heads, 1);
        assert_eq!(class.head_dim, 20);
        assert_eq!(class.window_tokens, 0);
        assert_eq!(class.layers(), vec![0, 1]);
    }

    fn mla_sidecar_with(class_fragment: &str) -> String {
        format!(
            r#"{{"name": "deepseek-mla-fixture", "num_layers": 2, "classes": [{class_fragment}]}}"#
        )
    }

    #[test]
    fn refuses_mla_latent_classes_with_bad_required_fields() {
        let missing_split = mla_sidecar_with(
            r#"{"class": "mla-latent", "from": 0, "until": 2, "step": 1,
                "except": [], "kv_heads": 1, "head_dim": 20, "window_tokens": 0}"#,
        );
        let error = parse_layout_sidecar(&missing_split).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar mla-latent class latent_dim is missing or not a positive integer"
        );

        let mismatched_split = mla_sidecar_with(
            r#"{"class": "mla-latent", "from": 0, "until": 2, "step": 1,
                "except": [], "kv_heads": 1, "head_dim": 21, "window_tokens": 0,
                "latent_dim": 16, "rope_dim": 4}"#,
        );
        let error = parse_layout_sidecar(&mismatched_split).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar mla-latent record must be one vector of latent_dim + rope_dim elements"
        );

        let multi_vector = mla_sidecar_with(
            r#"{"class": "mla-latent", "from": 0, "until": 2, "step": 1,
                "except": [], "kv_heads": 2, "head_dim": 20, "window_tokens": 0,
                "latent_dim": 16, "rope_dim": 4}"#,
        );
        assert!(parse_layout_sidecar(&multi_vector).is_err());

        let windowed = mla_sidecar_with(
            r#"{"class": "mla-latent", "from": 0, "until": 2, "step": 1,
                "except": [], "kv_heads": 1, "head_dim": 20, "window_tokens": 1024,
                "latent_dim": 16, "rope_dim": 4}"#,
        );
        let error = parse_layout_sidecar(&windowed).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar window_tokens are only legal on windowed classes"
        );

        let zero_latent = mla_sidecar_with(
            r#"{"class": "mla-latent", "from": 0, "until": 2, "step": 1,
                "except": [], "kv_heads": 1, "head_dim": 4, "window_tokens": 0,
                "latent_dim": 0, "rope_dim": 4}"#,
        );
        assert!(parse_layout_sidecar(&zero_latent).is_err());
    }

    #[test]
    fn refuses_latent_fields_on_non_mla_classes() {
        let json = mla_sidecar_with(
            r#"{"class": "gqa-full", "from": 0, "until": 2, "step": 1,
                "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0,
                "latent_dim": 16}"#,
        );
        let error = parse_layout_sidecar(&json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar latent_dim/rope_dim are only legal on mla-latent classes"
        );
    }

    fn gqa_sidecar_with_rope(rope_fragment: &str) -> String {
        format!(
            r#"{{"name": "rope-fixture", "num_layers": 2, "classes": [
                {{"class": "gqa-full", "from": 0, "until": 2, "step": 1,
                  "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0,
                  "rope": {rope_fragment}}}
            ]}}"#
        )
    }

    #[test]
    fn refuses_a_class_without_a_rope_object() {
        let json = r#"{
            "name": "no-rope",
            "num_layers": 2,
            "classes": [
                {"class": "gqa-full", "from": 0, "until": 2, "step": 1,
                 "except": [], "kv_heads": 4, "head_dim": 64, "window_tokens": 0}
            ]
        }"#;
        let error = parse_layout_sidecar(json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar class rope is missing"
        );
    }

    #[test]
    fn refuses_bad_rope_fields() {
        // Missing / out-of-bounds freq_base.
        let error = parse_layout_sidecar(&gqa_sidecar_with_rope(
            r#"{"dimension_count": 64, "scaling": "none", "convention": "neox"}"#,
        ))
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar class rope freq_base is missing or outside the supported bounds"
        );
        for bad_base in ["1.0", "0.5", "-1e6", "\"10000\""] {
            let json = gqa_sidecar_with_rope(&format!(
                r#"{{"freq_base": {bad_base}, "dimension_count": 64, "scaling": "none", "convention": "neox"}}"#
            ));
            assert!(parse_layout_sidecar(&json).is_err(), "base {bad_base}");
        }
        // dimension_count odd, zero, or wider than head_dim.
        for bad_count in ["0", "3", "128"] {
            let json = gqa_sidecar_with_rope(&format!(
                r#"{{"freq_base": 500000.0, "dimension_count": {bad_count}, "scaling": "none", "convention": "neox"}}"#
            ));
            let error = parse_layout_sidecar(&json).unwrap_err();
            assert_eq!(
                error.to_string(),
                "prefill layout sidecar class rope dimension_count is outside the v2 bounds"
            );
        }
        // Non-canonical scaling labels: unknown type, non-shortest factor,
        // missing factor.
        for bad_scaling in ["dynamic:32", "yarn:32.0", "yarn", "yarn:", "NONE"] {
            let json = gqa_sidecar_with_rope(&format!(
                r#"{{"freq_base": 500000.0, "dimension_count": 64, "scaling": "{bad_scaling}", "convention": "neox"}}"#
            ));
            let error = parse_layout_sidecar(&json).unwrap_err();
            assert_eq!(
                error.to_string(),
                "prefill layout sidecar class rope scaling is missing or not a canonical label",
                "scaling {bad_scaling}"
            );
        }
        // Convention outside the closed set.
        let json = gqa_sidecar_with_rope(
            r#"{"freq_base": 500000.0, "dimension_count": 64, "scaling": "none", "convention": "gpt-j"}"#,
        );
        let error = parse_layout_sidecar(&json).unwrap_err();
        assert_eq!(
            error.to_string(),
            "prefill layout sidecar class rope convention is missing or outside the closed set"
        );
    }

    #[test]
    fn sidecars_differing_only_in_freq_base_derive_different_identities() {
        let base = parse_layout_sidecar(&gqa_sidecar_with_rope(
            r#"{"freq_base": 500000.0, "dimension_count": 64, "scaling": "yarn:32", "convention": "neox"}"#,
        ))
        .unwrap();
        let changed = parse_layout_sidecar(&gqa_sidecar_with_rope(
            r#"{"freq_base": 1000000.0, "dimension_count": 64, "scaling": "yarn:32", "convention": "neox"}"#,
        ))
        .unwrap();
        assert_ne!(
            base.classes[0].rope_freq_base_bits,
            changed.classes[0].rope_freq_base_bits
        );
        let input = crate::prefill::PortablePrefillDescriptorInputV2 {
            model_sha256: [1; 32],
            adapter_sha256: [2; 32],
            tokenizer_sha256: [3; 32],
            chat_template_sha256: [4; 32],
            context_policy_sha256: [5; 32],
            model_revision: "model@revision".into(),
            tokenizer_revision: "tokenizer@revision".into(),
            producer_engine_abi: "vllm-0.21".into(),
            consumer_engine_abi: "ferrite-v1".into(),
            portable_abi: "canonical-kv-v2".into(),
            compute_precision: "float16".into(),
            kv_precision: "float16".into(),
            weight_precision: "q4_k_m".into(),
            cached_token_count: 1_024,
            max_context_tokens: 32_768,
            layout_name: "derived:rope-fixture".into(),
            transform: None,
            prerope_kernel_pin: None,
        };
        let base_descriptor =
            crate::prefill::derive_portable_prefill_descriptor_v2_from_layout(&input, &base)
                .unwrap();
        let changed_descriptor =
            crate::prefill::derive_portable_prefill_descriptor_v2_from_layout(&input, &changed)
                .unwrap();
        assert_ne!(
            base_descriptor.family.engine_cache_abi,
            changed_descriptor.family.engine_cache_abi
        );
    }
}
