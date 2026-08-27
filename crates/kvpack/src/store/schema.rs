use kvpack_core::Id32;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

use crate::{StoreConfig, StoreError};

use super::CATALOG_SCHEMA_VERSION;

const MIGRATION_1: &str =
    "CREATE TABLE tenants(namespace BLOB PRIMARY KEY CHECK(length(namespace)=32), operator_id_hash BLOB NOT NULL, key_epoch INTEGER NOT NULL, catalog_epoch INTEGER NOT NULL, quota_bytes INTEGER NOT NULL, durable_bytes INTEGER NOT NULL DEFAULT 0, reserved_bytes INTEGER NOT NULL DEFAULT 0);
     CREATE TABLE uploads(tenant BLOB NOT NULL, idempotency_key BLOB NOT NULL CHECK(length(idempotency_key)=32), state TEXT NOT NULL, reserved_bytes INTEGER NOT NULL, expected_bytes INTEGER NOT NULL, manifest_id BLOB, created_ns INTEGER NOT NULL, updated_ns INTEGER NOT NULL, PRIMARY KEY(tenant,idempotency_key));
     CREATE TABLE chunks(tenant BLOB NOT NULL, object_key BLOB NOT NULL CHECK(length(object_key)=32), chunk_id BLOB NOT NULL CHECK(length(chunk_id)=32), object_digest BLOB NOT NULL CHECK(length(object_digest)=32), key_epoch INTEGER NOT NULL, plaintext_bytes INTEGER NOT NULL, object_bytes INTEGER NOT NULL, refcount INTEGER NOT NULL DEFAULT 0, location_state TEXT NOT NULL, last_access_epoch INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(tenant,object_key), UNIQUE(tenant,chunk_id));
     CREATE TABLE manifests(tenant BLOB NOT NULL, manifest_id BLOB NOT NULL CHECK(length(manifest_id)=32), key_epoch INTEGER NOT NULL, catalog_epoch INTEGER NOT NULL, file_bytes INTEGER NOT NULL, restored_bytes INTEGER NOT NULL, semantic_id BLOB NOT NULL, family_id BLOB NOT NULL, token_count INTEGER NOT NULL, parent_id BLOB, parent_depth INTEGER NOT NULL, published_ns INTEGER NOT NULL, PRIMARY KEY(tenant,manifest_id));
     CREATE TABLE manifest_chunks(tenant BLOB NOT NULL, manifest_id BLOB NOT NULL, state_layer INTEGER NOT NULL, state_name TEXT NOT NULL, ordinal INTEGER NOT NULL, object_key BLOB NOT NULL, PRIMARY KEY(tenant,manifest_id,state_layer,state_name,ordinal));
     CREATE TABLE locations(tenant BLOB NOT NULL, object_kind TEXT NOT NULL, object_id BLOB NOT NULL, tier TEXT NOT NULL, state TEXT NOT NULL, locator TEXT NOT NULL, PRIMARY KEY(tenant,object_kind,object_id,tier));
     CREATE TABLE prefix_checkpoints(tenant BLOB NOT NULL, prefix_id BLOB NOT NULL, semantic_id BLOB NOT NULL, family_id BLOB NOT NULL, token_count INTEGER NOT NULL, manifest_id BLOB NOT NULL, exact_final INTEGER NOT NULL, PRIMARY KEY(tenant,prefix_id,semantic_id,family_id));
     CREATE INDEX prefix_lookup ON prefix_checkpoints(tenant,semantic_id,family_id,token_count DESC);
     CREATE TABLE pins(tenant BLOB NOT NULL, pin_id BLOB NOT NULL, object_key BLOB NOT NULL, owner_pid INTEGER NOT NULL, owner_start BLOB NOT NULL, created_ns INTEGER NOT NULL, PRIMARY KEY(tenant,pin_id));
     CREATE TABLE grants(tenant BLOB NOT NULL, capability BLOB NOT NULL, peer_id BLOB NOT NULL, plan_id BLOB NOT NULL, resource_bytes INTEGER NOT NULL, state TEXT NOT NULL, created_ns INTEGER NOT NULL, PRIMARY KEY(tenant,capability));
     CREATE TABLE tombstones(tenant BLOB NOT NULL, object_kind TEXT NOT NULL, object_id BLOB NOT NULL, catalog_epoch INTEGER NOT NULL, created_ns INTEGER NOT NULL, PRIMARY KEY(tenant,object_kind,object_id));
     CREATE TABLE write_tickets(tenant BLOB NOT NULL, ticket_id BLOB NOT NULL, bucket_start_ns INTEGER NOT NULL, reserved_bytes INTEGER NOT NULL, consumed_bytes INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(tenant,ticket_id));
     CREATE INDEX write_ticket_window ON write_tickets(tenant,bucket_start_ns);
     CREATE TABLE access_epochs(tenant BLOB NOT NULL, object_kind TEXT NOT NULL, object_id BLOB NOT NULL, epoch INTEGER NOT NULL, score REAL NOT NULL, PRIMARY KEY(tenant,object_kind,object_id));";

const MIGRATION_2: &str =
    "ALTER TABLE uploads ADD COLUMN next_chunk_ordinal INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE uploads ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE uploads ADD COLUMN abort_reason TEXT;
     ALTER TABLE chunks ADD COLUMN created_ns INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE chunks ADD COLUMN last_access_ns INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE chunks ADD COLUMN retention_segment TEXT NOT NULL DEFAULT 'PROBATIONARY';
     ALTER TABLE chunks ADD COLUMN frequency_estimate INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE manifests ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;
     CREATE TABLE leases(tenant BLOB NOT NULL,lease_id BLOB NOT NULL CHECK(length(lease_id)=32),object_kind TEXT NOT NULL,object_id BLOB NOT NULL,owner_id BLOB NOT NULL,state TEXT NOT NULL,expires_ns INTEGER NOT NULL,created_ns INTEGER NOT NULL,PRIMARY KEY(tenant,lease_id));
     CREATE INDEX lease_object ON leases(tenant,object_kind,object_id,state,expires_ns);
     CREATE TABLE upload_chunks(tenant BLOB NOT NULL,idempotency_key BLOB NOT NULL,ordinal INTEGER NOT NULL,object_key BLOB NOT NULL,object_digest BLOB NOT NULL,verified_ns INTEGER NOT NULL,PRIMARY KEY(tenant,idempotency_key,ordinal));
     CREATE TABLE quarantine_entries(tenant BLOB NOT NULL,entry_id BLOB NOT NULL CHECK(length(entry_id)=32),object_kind TEXT NOT NULL,object_id BLOB,path_token TEXT NOT NULL,file_bytes INTEGER NOT NULL,created_ns INTEGER NOT NULL,expires_ns INTEGER NOT NULL,reason TEXT NOT NULL,PRIMARY KEY(tenant,entry_id));
     CREATE INDEX quarantine_expiry ON quarantine_entries(tenant,expires_ns,created_ns);
     CREATE TABLE generations(tenant BLOB NOT NULL,scope TEXT NOT NULL,scope_id BLOB NOT NULL,generation INTEGER NOT NULL,state TEXT NOT NULL,updated_ns INTEGER NOT NULL,PRIMARY KEY(tenant,scope,scope_id));
     CREATE TABLE policy_objects(tenant BLOB NOT NULL,object_key BLOB NOT NULL,frequency INTEGER NOT NULL DEFAULT 0,segment TEXT NOT NULL,score INTEGER NOT NULL DEFAULT 0,last_access_ns INTEGER NOT NULL DEFAULT 0,last_persisted_epoch INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(tenant,object_key));
     CREATE INDEX policy_victim ON policy_objects(tenant,segment,score,last_access_ns);
     CREATE TABLE policy_meta(tenant BLOB NOT NULL,key TEXT NOT NULL,value INTEGER NOT NULL,updated_ns INTEGER NOT NULL,PRIMARY KEY(tenant,key));
     CREATE TABLE audit_state(tenant BLOB NOT NULL,stream TEXT NOT NULL,next_sequence INTEGER NOT NULL,last_flushed_ns INTEGER NOT NULL,PRIMARY KEY(tenant,stream));";

const MIGRATION_3: &str = "UPDATE uploads SET state='RESERVED' WHERE state='WRITE_RESERVED';
     UPDATE uploads SET state='VERIFIED' WHERE state IN ('CHUNKS_VERIFIED','MANIFEST_VERIFIED');
     DROP INDEX policy_victim;
     ALTER TABLE policy_objects RENAME TO policy_objects_v2;
     CREATE TABLE policy_objects(tenant BLOB NOT NULL,object_key BLOB NOT NULL,frequency INTEGER NOT NULL DEFAULT 0,segment TEXT NOT NULL,score INTEGER NOT NULL DEFAULT 0,last_access_ns INTEGER NOT NULL DEFAULT 0,last_persisted_epoch INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(tenant,object_key));
     INSERT INTO policy_objects(tenant,object_key,frequency,segment,score,last_access_ns,last_persisted_epoch) SELECT tenant,object_key,frequency,segment,CAST(score AS INTEGER),last_access_ns,last_persisted_epoch FROM policy_objects_v2;
     DROP TABLE policy_objects_v2;
     CREATE INDEX policy_victim ON policy_objects(tenant,segment,score,last_access_ns);";

const MIGRATION_4: &str =
    "ALTER TABLE uploads ADD COLUMN intent_digest BLOB NOT NULL DEFAULT X'0000000000000000000000000000000000000000000000000000000000000000' CHECK(length(intent_digest)=32);
     UPDATE uploads SET generation=1 WHERE generation=0;
     UPDATE manifests SET generation=1 WHERE generation=0;";

const MIGRATION_5: &str =
    "ALTER TABLE tenants ADD COLUMN staging_quota_bytes INTEGER NOT NULL DEFAULT 0;
     UPDATE tenants SET staging_quota_bytes=quota_bytes WHERE staging_quota_bytes=0;";

const MIGRATION_6: &str =
    "ALTER TABLE tenants ADD COLUMN minimum_readable_key_epoch INTEGER NOT NULL DEFAULT 1;";

const MIGRATION_7: &str =
    "ALTER TABLE chunks RENAME TO chunks_v6;
     CREATE TABLE chunks(tenant BLOB NOT NULL, object_key BLOB NOT NULL CHECK(length(object_key)=32), chunk_id BLOB NOT NULL CHECK(length(chunk_id)=32), object_digest BLOB NOT NULL CHECK(length(object_digest)=32), key_epoch INTEGER NOT NULL, plaintext_bytes INTEGER NOT NULL, object_bytes INTEGER NOT NULL, refcount INTEGER NOT NULL DEFAULT 0, location_state TEXT NOT NULL, last_access_epoch INTEGER NOT NULL DEFAULT 0, created_ns INTEGER NOT NULL DEFAULT 0, last_access_ns INTEGER NOT NULL DEFAULT 0, retention_segment TEXT NOT NULL DEFAULT 'PROBATIONARY', frequency_estimate INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(tenant,object_key), UNIQUE(tenant,chunk_id,key_epoch));
     INSERT INTO chunks(tenant,object_key,chunk_id,object_digest,key_epoch,plaintext_bytes,object_bytes,refcount,location_state,last_access_epoch,created_ns,last_access_ns,retention_segment,frequency_estimate) SELECT tenant,object_key,chunk_id,object_digest,key_epoch,plaintext_bytes,object_bytes,refcount,location_state,last_access_epoch,created_ns,last_access_ns,retention_segment,frequency_estimate FROM chunks_v6;
     DROP TABLE chunks_v6;";

const MIGRATION_8: &str =
    "ALTER TABLE uploads ADD COLUMN seal_digest BLOB NOT NULL DEFAULT X'0000000000000000000000000000000000000000000000000000000000000000' CHECK(length(seal_digest)=32);
     CREATE TABLE remote_upload_fences(tenant BLOB NOT NULL,idempotency_key BLOB NOT NULL CHECK(length(idempotency_key)=32),manifest_id BLOB NOT NULL CHECK(length(manifest_id)=32),owner_id BLOB NOT NULL CHECK(length(owner_id)=32),attempt_epoch INTEGER NOT NULL,generation INTEGER NOT NULL,created_ns INTEGER NOT NULL,PRIMARY KEY(tenant,idempotency_key));
     CREATE TABLE remote_mutations(tenant BLOB NOT NULL,nonce BLOB NOT NULL CHECK(length(nonce)=32),request_id BLOB NOT NULL CHECK(length(request_id)=32),operation TEXT NOT NULL,scope_id BLOB NOT NULL CHECK(length(scope_id)=32),intent_digest BLOB NOT NULL CHECK(length(intent_digest)=32),first_seen_ns INTEGER NOT NULL,last_seen_ns INTEGER NOT NULL,PRIMARY KEY(tenant,nonce));
     CREATE TABLE source_leases(tenant BLOB NOT NULL,lease_id BLOB NOT NULL CHECK(length(lease_id)=32),manifest_id BLOB NOT NULL CHECK(length(manifest_id)=32),owner_id BLOB NOT NULL CHECK(length(owner_id)=32),owner_incarnation BLOB NOT NULL CHECK(length(owner_incarnation)=32),authority_term INTEGER NOT NULL,generation INTEGER NOT NULL,state TEXT NOT NULL,granted_ns INTEGER NOT NULL,maximum_duration_ns INTEGER NOT NULL,expires_ns INTEGER NOT NULL,PRIMARY KEY(tenant,lease_id));
     CREATE TABLE source_lease_objects(tenant BLOB NOT NULL,lease_id BLOB NOT NULL,object_kind TEXT NOT NULL,object_id BLOB NOT NULL CHECK(length(object_id)=32),ordinal INTEGER NOT NULL,PRIMARY KEY(tenant,lease_id,object_kind,object_id));
     CREATE INDEX source_lease_object ON source_lease_objects(tenant,object_kind,object_id,lease_id);
     CREATE INDEX source_lease_state ON source_leases(tenant,state,expires_ns);";

const MIGRATION_9: &str =
    "ALTER TABLE audit_state ADD COLUMN backpressure_events INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE audit_state ADD COLUMN delivery_failures INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE audit_state ADD COLUMN retention_pruned_records INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE audit_state ADD COLUMN lost_records INTEGER NOT NULL DEFAULT 0;
     CREATE TABLE audit_outbox(tenant BLOB NOT NULL,stream TEXT NOT NULL CHECK(stream='publication'),sequence INTEGER NOT NULL,event_kind TEXT NOT NULL CHECK(event_kind IN ('reserved','receiving','verified','published','aborted','quarantined','tombstoned','collected')),object_kind TEXT NOT NULL CHECK(object_kind IN ('upload','prefix','manifest','chunk','quarantine')),object_id BLOB NOT NULL CHECK(length(object_id)=32),generation INTEGER NOT NULL,occurred_ns INTEGER NOT NULL,delivered_ns INTEGER,delivery_attempts INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(tenant,stream,sequence));
     CREATE UNIQUE INDEX audit_event_identity ON audit_outbox(tenant,stream,event_kind,object_kind,object_id,generation);
     CREATE INDEX audit_delivery_queue ON audit_outbox(tenant,stream,delivered_ns,sequence);";

const MIGRATION_10: &str =
    "ALTER TABLE uploads ADD COLUMN session_token INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE uploads ADD COLUMN lease_expires_ns INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE uploads ADD COLUMN boundary_token_id INTEGER;
     ALTER TABLE uploads ADD COLUMN provenance_source_ns INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE uploads ADD COLUMN provenance_clock_offset_ns INTEGER;
     ALTER TABLE uploads ADD COLUMN provenance_quiesced INTEGER NOT NULL DEFAULT 0;";

/// M6 fidelity ladder: per-chunk-object fidelity rung.  0 resident-fp16
/// (today's behavior), 1 rest-quantized, 2 tombstone (bytes dropped, chained
/// key + catalog row retained for guided recompute).  Default 0 keeps every
/// pre-existing row on the resident-fp16 rung unchanged.
const MIGRATION_11: &str =
    "ALTER TABLE chunks ADD COLUMN fidelity_rung INTEGER NOT NULL DEFAULT 0 CHECK(fidelity_rung IN (0,1,2));";

pub(super) fn migrate(connection: &mut Connection) -> Result<(), StoreError> {
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "synchronous", "FULL")?;
    connection.pragma_update(None, "foreign_keys", "ON")?;
    connection.pragma_update(None, "trusted_schema", "OFF")?;
    connection.busy_timeout(std::time::Duration::from_secs(5))?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_ns INTEGER NOT NULL);",
    )?;
    let versions = {
        let mut statement =
            transaction.prepare("SELECT version FROM schema_migrations ORDER BY version")?;
        let values = statement
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        values
    };
    if versions
        .iter()
        .enumerate()
        .any(|(index, version)| *version != index as i64 + 1 || *version > CATALOG_SCHEMA_VERSION)
    {
        return Err(StoreError::State(
            "catalog contains an unknown or newer schema migration",
        ));
    }
    let applied = versions.last().copied().unwrap_or(0);
    for version in applied + 1..=CATALOG_SCHEMA_VERSION {
        apply_migration(&transaction, version)?;
        transaction.execute(
            "INSERT INTO schema_migrations(version,applied_ns) VALUES(?1,?2)",
            params![version, now_ns()],
        )?;
    }
    transaction.commit()?;
    Ok(())
}

fn apply_migration(transaction: &Transaction<'_>, version: i64) -> Result<(), StoreError> {
    match version {
        1 => transaction.execute_batch(MIGRATION_1)?,
        2 => transaction.execute_batch(MIGRATION_2)?,
        3 => transaction.execute_batch(MIGRATION_3)?,
        4 => transaction.execute_batch(MIGRATION_4)?,
        5 => transaction.execute_batch(MIGRATION_5)?,
        6 => transaction.execute_batch(MIGRATION_6)?,
        7 => transaction.execute_batch(MIGRATION_7)?,
        8 => transaction.execute_batch(MIGRATION_8)?,
        9 => transaction.execute_batch(MIGRATION_9)?,
        10 => transaction.execute_batch(MIGRATION_10)?,
        11 => transaction.execute_batch(MIGRATION_11)?,
        _ => {
            return Err(StoreError::State(
                "catalog contains an unknown or newer schema migration",
            ));
        }
    }
    Ok(())
}

pub(super) fn register_tenant(
    connection: &mut Connection,
    config: &StoreConfig,
    namespace: &Id32,
) -> Result<(), StoreError> {
    use sha2::{Digest, Sha256};
    let operator_hash: Id32 = Sha256::digest(&config.operator_tenant_id).into();
    let existing: Option<(u64, u64, u64)> = connection
        .query_row(
            "SELECT key_epoch,minimum_readable_key_epoch,catalog_epoch FROM tenants WHERE namespace=?1",
            [namespace.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    if existing.is_some_and(|(active, minimum, _)| {
        config.key_epoch < active || config.minimum_readable_key_epoch < minimum
    }) {
        return Err(StoreError::State(
            "key epochs cannot move backward in an existing catalog",
        ));
    }
    // The catalog epoch fences every remote capability; a stale-config
    // restart must not roll it backward either (audit N2).
    if existing.is_some_and(|(_, _, catalog)| config.catalog_epoch < catalog) {
        return Err(StoreError::State(
            "catalog epoch cannot move backward in an existing catalog",
        ));
    }
    connection.execute("INSERT INTO tenants(namespace,operator_id_hash,key_epoch,minimum_readable_key_epoch,catalog_epoch,quota_bytes,staging_quota_bytes) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(namespace) DO UPDATE SET key_epoch=excluded.key_epoch,minimum_readable_key_epoch=excluded.minimum_readable_key_epoch,catalog_epoch=excluded.catalog_epoch,quota_bytes=excluded.quota_bytes,staging_quota_bytes=excluded.staging_quota_bytes", params![namespace.as_slice(), operator_hash.as_slice(), config.key_epoch, config.minimum_readable_key_epoch, config.catalog_epoch, config.quota_bytes, config.staging_quota_bytes])?;
    connection.execute(
        "INSERT OR IGNORE INTO audit_state(tenant,stream,next_sequence,last_flushed_ns) VALUES(?1,'publication',1,0)",
        [namespace.as_slice()],
    )?;
    Ok(())
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_one_catalog_upgrades_through_every_ordered_migration() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_ns INTEGER NOT NULL);",
            )
            .unwrap();
        connection.execute_batch(MIGRATION_1).unwrap();
        connection
            .execute(
                "INSERT INTO schema_migrations(version,applied_ns) VALUES(1,0)",
                [],
            )
            .unwrap();
        connection.execute("INSERT INTO tenants(namespace,operator_id_hash,key_epoch,catalog_epoch,quota_bytes,durable_bytes,reserved_bytes) VALUES(?1,?2,1,1,1024,0,3)", params![[1u8; 32].as_slice(), [2u8; 32].as_slice()]).unwrap();
        for (value, state) in [
            (3u8, "WRITE_RESERVED"),
            (4, "CHUNKS_VERIFIED"),
            (5, "MANIFEST_VERIFIED"),
        ] {
            connection.execute("INSERT INTO uploads(tenant,idempotency_key,state,reserved_bytes,expected_bytes,created_ns,updated_ns) VALUES(?1,?2,?3,1,1,0,0)", params![[1u8; 32].as_slice(), [value; 32].as_slice(), state]).unwrap();
        }

        migrate(&mut connection).unwrap();
        let versions = {
            let mut statement = connection
                .prepare("SELECT version FROM schema_migrations ORDER BY version")
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, i64>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(versions, (1..=CATALOG_SCHEMA_VERSION).collect::<Vec<_>>());
        let added_column: u64 = connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name='retention_segment'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(added_column, 1);
        let score_type: String = connection
            .query_row(
                "SELECT type FROM pragma_table_info('policy_objects') WHERE name='score'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(score_type, "INTEGER");
        let intent_digest_length: u64 = connection
            .query_row(
                "SELECT length(intent_digest) FROM uploads ORDER BY idempotency_key LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(intent_digest_length, 32);
        let staging_quota: u64 = connection
            .query_row("SELECT staging_quota_bytes FROM tenants", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(staging_quota, 1024);
        let audit_columns: u64 = connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('audit_state') WHERE name IN ('backpressure_events','delivery_failures','retention_pruned_records','lost_records')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(audit_columns, 4);
        let audit_outbox_exists: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type='table' AND name='audit_outbox')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(audit_outbox_exists);
        let minimum_readable_key_epoch: u64 = connection
            .query_row(
                "SELECT minimum_readable_key_epoch FROM tenants",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(minimum_readable_key_epoch, 1);
        let chunk_unique_columns = {
            let mut statement = connection
                .prepare("SELECT name FROM pragma_index_info((SELECT name FROM pragma_index_list('chunks') WHERE \"unique\"=1 LIMIT 1)) ORDER BY seqno")
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(chunk_unique_columns, ["tenant", "chunk_id", "key_epoch"]);
        let minimum_generation: u64 = connection
            .query_row("SELECT MIN(generation) FROM uploads", [], |row| row.get(0))
            .unwrap();
        assert_eq!(minimum_generation, 1);
        let states = {
            let mut statement = connection
                .prepare("SELECT state FROM uploads ORDER BY idempotency_key")
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(states, ["RESERVED", "VERIFIED", "VERIFIED"]);
        let remote_tables: u64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name IN ('remote_upload_fences','remote_mutations','source_leases','source_lease_objects')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remote_tables, 4);
        let fidelity_rung_column: u64 = connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('chunks') WHERE name='fidelity_rung'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(fidelity_rung_column, 1);
    }

    #[test]
    fn fidelity_rung_defaults_to_resident_fp16_and_rejects_unknown_rungs() {
        let mut connection = Connection::open_in_memory().unwrap();
        migrate(&mut connection).unwrap();
        connection
            .execute(
                "INSERT INTO chunks(tenant,object_key,chunk_id,object_digest,key_epoch,plaintext_bytes,object_bytes,refcount,location_state) VALUES(?1,?2,?3,?4,1,8,4096,0,'AVAILABLE')",
                params![[1u8; 32].as_slice(), [2u8; 32].as_slice(), [3u8; 32].as_slice(), [4u8; 32].as_slice()],
            )
            .unwrap();
        let default_rung: u64 = connection
            .query_row("SELECT fidelity_rung FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(default_rung, 0);
        let invalid = connection.execute(
            "INSERT INTO chunks(tenant,object_key,chunk_id,object_digest,key_epoch,plaintext_bytes,object_bytes,refcount,location_state,fidelity_rung) VALUES(?1,?2,?3,?4,1,8,4096,0,'AVAILABLE',3)",
            params![[1u8; 32].as_slice(), [5u8; 32].as_slice(), [6u8; 32].as_slice(), [7u8; 32].as_slice()],
        );
        assert!(invalid.is_err());
    }
}
