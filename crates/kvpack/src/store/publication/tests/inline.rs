#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use kvpack_core::{
        AuxiliaryInputId, CacheKind, Codec, DType, FamilyState, Layout, RepresentationFamilyId,
        RepresentationMode, SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
    };

    use crate::{
        ExportCutPolicy, ExportDeclaration, ExportSession, ExportStateDeclaration, StoreConfig,
        WritePolicy,
    };

    use super::*;

    const IMMUTABLE_PHASES: [ImmutableFaultPhase; 6] = [
        ImmutableFaultPhase::Create,
        ImmutableFaultPhase::Write,
        ImmutableFaultPhase::FileSync,
        ImmutableFaultPhase::NoReplace,
        ImmutableFaultPhase::TargetDirectorySync,
        ImmutableFaultPhase::PartialDirectorySync,
    ];

    struct Fixture {
        temp: tempfile::TempDir,
        config: StoreConfig,
        store: Arc<LocalStore>,
    }

    fn id(value: u8) -> Id32 {
        [value; 32]
    }

    fn semantic() -> SemanticModelId {
        SemanticModelId {
            weights_config: id(1),
            adapters: id(2),
            tokenizer_template: id(3),
            position_semantics: id(4),
            qualified_math: id(5),
        }
    }

    fn family() -> RepresentationFamilyId {
        RepresentationFamilyId {
            engine_cache_abi: id(6),
            mode: RepresentationMode::Native,
            page_size_tokens: 256,
            topology: id(7),
            shard_map: id(8),
            states: vec![FamilyState {
                key: StateKey::new(0, "k"),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::U8,
                codec: Codec::Raw,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token: 8,
                dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(8)],
                dependencies: vec![],
            }],
        }
    }

    fn declaration() -> ExportDeclaration {
        ExportDeclaration {
            semantic_model: semantic(),
            input_tokens: vec![10, 20, 30, 40],
            auxiliary_inputs: vec![AuxiliaryInputId {
                type_id: id(30),
                value_id: id(31),
            }],
            family: family(),
            states: vec![ExportStateDeclaration {
                key: StateKey::new(0, "k"),
                strides: vec![8, 1],
                atomic_group: 1,
            }],
        }
    }

    fn fixture(name: &[u8]) -> Fixture {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        crate::create_store_key_random(&key_path, temp.path()).unwrap();
        let config = StoreConfig {
            object_root: temp.path().join("objects"),
            catalog_path: temp.path().join("catalog/catalog.sqlite"),
            operator_tenant_id: name.to_vec(),
            key_epoch: 1,
            minimum_readable_key_epoch: 1,
            catalog_epoch: 1,
            quota_bytes: 1 << 30,
            staging_quota_bytes: 1 << 30,
            endurance_bytes_per_five_minutes: 1 << 30,
        };
        let store = Arc::new(
            LocalStore::open(
                config.clone(),
                crate::load_store_key(&key_path, temp.path()).unwrap(),
            )
            .unwrap(),
        );
        Fixture {
            temp,
            config,
            store,
        }
    }

    fn begin(store: Arc<LocalStore>, idempotency: Id32) -> ExportSession {
        let family = family();
        ExportSession::begin(
            store,
            declaration(),
            ExportCutPolicy::production_v1(),
            WritePolicy::exact_qualified(idempotency, semantic(), &family).unwrap(),
        )
        .unwrap()
    }

    fn complete_state(session: &mut ExportSession) {
        let mut source = Cursor::new(vec![7; 32]);
        session
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut source)
            .unwrap();
    }

    fn arm(store: &LocalStore, point: DurabilityFaultPoint) {
        *store.durability_fault.lock().unwrap() = Some(point);
    }

    fn assert_injected<T>(result: Result<T, StoreError>) {
        assert!(matches!(
            result,
            Err(StoreError::Io {
                op: "injected durability fault",
                ..
            })
        ));
    }

    fn assert_hidden(store: &LocalStore, idempotency: Id32, expected_manifests: u64) {
        let stat = store.stat().unwrap();
        assert_eq!(stat.reserved_bytes, 0);
        assert_eq!(stat.manifests, expected_manifests);
        let connection = store.lock_catalog().unwrap();
        let state: String = connection
            .query_row(
                "SELECT state FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![store.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "ABORTED");
        let prefixes: u64 = connection
            .query_row("SELECT COUNT(*) FROM prefix_checkpoints", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(prefixes, expected_manifests);
        let published_audit: u64 = connection
            .query_row(
                "SELECT COUNT(*) FROM audit_outbox WHERE tenant=?1 AND stream='publication' AND event_kind='published'",
                [store.tenant_namespace.as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(published_audit, expected_manifests);
    }

    fn reopen(fixture: Fixture) -> Fixture {
        let Fixture {
            temp,
            config,
            store,
        } = fixture;
        drop(store);
        let key = crate::load_store_key(&temp.path().join("keys/root.key"), temp.path()).unwrap();
        let reopened = Arc::new(LocalStore::open(config.clone(), key).unwrap());
        assert_eq!(
            fs::read_dir(reopened.config.object_root.join("partials"))
                .unwrap()
                .count(),
            0
        );
        Fixture {
            temp,
            config,
            store: reopened,
        }
    }

    #[test]
    fn every_chunk_immutable_barrier_aborts_without_a_visible_cut() {
        for phase in IMMUTABLE_PHASES {
            let fixture = fixture(format!("chunk-fault-{phase:?}").as_bytes());
            let upload = id(80);
            let mut session = begin(Arc::clone(&fixture.store), upload);
            arm(
                &fixture.store,
                DurabilityFaultPoint::Immutable(DurableObjectKind::Chunk, phase),
            );
            let mut source = Cursor::new(vec![7; 32]);
            assert_injected(
                session
                    .next_state(StateKey::new(0, "k"))
                    .unwrap()
                    .write_source(&mut source),
            );
            assert!(matches!(session.commit(), Err(StoreError::Poisoned(_))));
            assert_hidden(&fixture.store, upload, 0);
            assert!(fixture.store.durability_fault.lock().unwrap().is_none());
            let reopened = reopen(fixture);
            assert_hidden(&reopened.store, upload, 0);
        }
    }

    #[test]
    fn every_manifest_immutable_barrier_aborts_without_a_visible_cut() {
        for phase in IMMUTABLE_PHASES {
            let fixture = fixture(format!("manifest-fault-{phase:?}").as_bytes());
            let upload = id(81);
            let mut session = begin(Arc::clone(&fixture.store), upload);
            complete_state(&mut session);
            arm(
                &fixture.store,
                DurabilityFaultPoint::Immutable(DurableObjectKind::Manifest, phase),
            );
            assert_injected(session.commit());
            assert_hidden(&fixture.store, upload, 0);
            assert!(fixture.store.durability_fault.lock().unwrap().is_none());
            let reopened = reopen(fixture);
            assert_hidden(&reopened.store, upload, 0);
        }
    }

    #[test]
    fn every_upload_manifest_barrier_is_cancelable_and_undiscoverable() {
        for phase in IMMUTABLE_PHASES {
            let fixture = fixture(format!("upload-manifest-fault-{phase:?}").as_bytes());
            let mut source = begin(Arc::clone(&fixture.store), id(82));
            complete_state(&mut source);
            let published = source.commit().unwrap();
            let manifest_id = published.exact_final.manifest_id;
            let context = kvpack_core::ValidationContext::default();
            let pack = fixture
                .store
                .read_authenticated_manifest_object(&manifest_id, &context)
                .unwrap();
            let upload = id(83);
            fixture
                .store
                .begin_authenticated_import(&upload, &manifest_id, 1 << 20, 2)
                .unwrap();
            arm(
                &fixture.store,
                DurabilityFaultPoint::Immutable(DurableObjectKind::UploadManifest, phase),
            );
            assert_injected(
                fixture
                    .store
                    .stage_authenticated_manifest(&upload, &pack, &context),
            );
            fixture.store.cancel_authenticated_import(&upload).unwrap();
            assert_hidden(&fixture.store, upload, 1);
            assert!(fixture.store.durability_fault.lock().unwrap().is_none());
            let reopened = reopen(fixture);
            assert_hidden(&reopened.store, upload, 1);
        }
    }

    #[test]
    fn catalog_begin_and_commit_faults_roll_back_visibility() {
        for point in [
            DurabilityFaultPoint::CatalogBegin,
            DurabilityFaultPoint::CatalogCommit,
        ] {
            let fixture = fixture(format!("catalog-fault-{point:?}").as_bytes());
            let upload = id(84);
            let mut session = begin(Arc::clone(&fixture.store), upload);
            complete_state(&mut session);
            arm(&fixture.store, point);
            assert_injected(session.commit());
            assert_hidden(&fixture.store, upload, 0);
            assert!(fixture.store.durability_fault.lock().unwrap().is_none());
            let reopened = reopen(fixture);
            assert_hidden(&reopened.store, upload, 0);
        }
    }
}
