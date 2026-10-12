use std::error::Error;
use std::fmt;

/// Version of the deterministic production capacity worksheet.
///
/// This is an operational schema and is independent from durable object and
/// service protocol versions.
pub const CAPACITY_WORKSHEET_SCHEMA_VERSION: u32 = 1;

pub const PRODUCTION_CAPACITY_RATIO_MILLIS: u64 = 1_000;
pub const PRODUCTION_DURABLE_TARGET_MILLIS: u64 = 750;
pub const PRODUCTION_QUARANTINE_MILLIS: u64 = 10;
pub const PRODUCTION_INTERNAL_TARGET_MILLIS: u64 = 750;
pub const PRODUCTION_WRITE_AMPLIFICATION_LIMIT_MILLIS: u64 = 1_300;

pub const PRODUCTION_GATEWAY_DATA_STREAMS: u64 = 4;
pub const PRODUCTION_GATEWAY_RECEIVE_BUFFERS_PER_STREAM: u64 = 2;
pub const PRODUCTION_GATEWAY_RECEIVE_BUFFER_BYTES: u64 = 4 * 1024 * 1024;
pub const PRODUCTION_GATEWAY_IN_FLIGHT_BYTES: u64 = PRODUCTION_GATEWAY_DATA_STREAMS
    * PRODUCTION_GATEWAY_RECEIVE_BUFFERS_PER_STREAM
    * PRODUCTION_GATEWAY_RECEIVE_BUFFER_BYTES;

pub const MAX_CAPACITY_CONCURRENCY: u64 = 1_000_000;
pub const MAX_CAPACITY_GATEWAY_INSTANCES: u64 = 65_536;
pub const MAX_CAPACITY_CATALOG_BYTES_PER_ROW: u64 = 1024 * 1024;
pub const MAX_CAPACITY_WRITE_AMPLIFICATION_MILLIS: u64 = 10_000;
pub const MAX_CAPACITY_NETWORK_UTILIZATION_MILLIS: u64 = 950;
pub const MAX_CAPACITY_BACKUP_GENERATIONS: u64 = 3_650;

/// Measured or conservatively bounded workload inputs for one deployment.
///
/// Byte rates are encoded object bytes unless a field explicitly says KV.
/// `maximum_agent_plan_bytes` is the largest authenticated
/// `ResourcePlan.total_bytes` admitted for any qualified cache family; it
/// already contains shadow, pinned source, scratch, staging, registered
/// transport, receive-window, and engine-margin charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityWorksheetInput {
    pub retained_unique_kv_bytes: u64,
    pub retained_object_overhead_bytes: u64,
    pub pinned_object_bytes: u64,
    pub catalog_row_count: u64,
    pub catalog_bytes_per_row: u64,
    pub catalog_fixed_bytes: u64,
    pub wal_reserve_bytes: u64,
    pub concurrent_uploads: u64,
    pub maximum_upload_bytes: u64,
    pub concurrent_restores: u64,
    pub maximum_agent_plan_bytes: u64,
    pub source_pins_per_restore: u64,
    pub source_fds_per_restore: u64,
    pub gateway_instances: u64,
    pub daily_unique_kv_bytes: u64,
    pub peak_five_minute_unique_kv_bytes: u64,
    pub peak_five_minute_new_object_bytes: u64,
    pub physical_write_amplification_millis: u64,
    pub device_write_budget_bytes_per_day: u64,
    pub peak_network_payload_bytes_per_second: u64,
    pub network_target_utilization_millis: u64,
    pub backup_bytes_per_generation: u64,
    pub retained_backup_generations: u64,
}

/// Fixed-policy capacity requirements derived without floating-point or
/// saturating arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityWorksheet {
    pub schema_version: u32,
    pub input: CapacityWorksheetInput,
    pub retained_object_bytes: u64,
    pub pinned_object_bytes: u64,
    pub evictable_object_bytes: u64,
    pub staging_quota_bytes: u64,
    pub store_quota_bytes: u64,
    pub quarantine_cap_bytes: u64,
    pub minimum_object_filesystem_bytes: u64,
    pub estimated_catalog_live_bytes: u64,
    pub wal_reserve_bytes: u64,
    pub catalog_backup_temporary_bytes: u64,
    pub minimum_internal_filesystem_bytes: u64,
    pub agent_admission_capacity_bytes: u64,
    pub gateway_transport_window_bytes: u64,
    pub admission_and_window_bytes: u64,
    pub maximum_active_grants: u64,
    pub required_source_pins: u64,
    pub required_source_fds: u64,
    pub endurance_bytes_per_five_minutes: u64,
    pub physical_write_bytes_per_five_minutes: u64,
    pub physical_write_bytes_per_day: u64,
    pub device_write_budget_bytes_per_day: u64,
    pub endurance_within_device_budget: bool,
    pub physical_write_amplification_within_v1_gate: bool,
    pub required_network_bits_per_second: u64,
    pub backup_retention_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityPlanningError {
    Invalid(&'static str),
    Overflow(&'static str),
}

impl fmt::Display for CapacityPlanningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid capacity input: {message}"),
            Self::Overflow(message) => {
                write!(formatter, "capacity calculation overflow: {message}")
            }
        }
    }
}

impl Error for CapacityPlanningError {}

impl CapacityWorksheet {
    /// Derive the production-v1 worksheet using the store's 75% retained-data
    /// target, 1% quarantine cap, fixed gateway windows, and 1.3x write gate.
    /// The result recommends capacities; it does not mutate store policy.
    pub fn production_v1(input: CapacityWorksheetInput) -> Result<Self, CapacityPlanningError> {
        validate(input)?;

        let retained_object_bytes = add(
            input.retained_unique_kv_bytes,
            input.retained_object_overhead_bytes,
            "retained object bytes",
        )?;
        if input.pinned_object_bytes > retained_object_bytes {
            return Err(CapacityPlanningError::Invalid(
                "pinned object bytes exceed retained object bytes",
            ));
        }
        let evictable_object_bytes = retained_object_bytes - input.pinned_object_bytes;
        let staging_quota_bytes = multiply(
            input.concurrent_uploads,
            input.maximum_upload_bytes,
            "staging quota",
        )?;
        let retained_target_quota = scale_ceil(
            retained_object_bytes,
            PRODUCTION_CAPACITY_RATIO_MILLIS,
            PRODUCTION_DURABLE_TARGET_MILLIS,
            "retained-data store quota",
        )?;
        let retained_plus_staging = add(
            retained_object_bytes,
            staging_quota_bytes,
            "retained plus staging bytes",
        )?;
        let store_quota_bytes = retained_target_quota.max(retained_plus_staging);
        let quarantine_cap_bytes = scale_floor(
            store_quota_bytes,
            PRODUCTION_QUARANTINE_MILLIS,
            PRODUCTION_CAPACITY_RATIO_MILLIS,
            "quarantine cap",
        )?;
        let minimum_object_filesystem_bytes = add(
            store_quota_bytes,
            quarantine_cap_bytes,
            "object filesystem capacity",
        )?;

        let catalog_rows_bytes = multiply(
            input.catalog_row_count,
            input.catalog_bytes_per_row,
            "catalog row estimate",
        )?;
        let estimated_catalog_live_bytes = add(
            input.catalog_fixed_bytes,
            catalog_rows_bytes,
            "live catalog estimate",
        )?;
        let catalog_backup_temporary_bytes = estimated_catalog_live_bytes;
        let internal_live_and_wal = add(
            estimated_catalog_live_bytes,
            input.wal_reserve_bytes,
            "catalog plus WAL reserve",
        )?;
        let internal_with_backup = add(
            internal_live_and_wal,
            catalog_backup_temporary_bytes,
            "catalog backup working set",
        )?;
        let minimum_internal_filesystem_bytes = scale_ceil(
            internal_with_backup,
            PRODUCTION_CAPACITY_RATIO_MILLIS,
            PRODUCTION_INTERNAL_TARGET_MILLIS,
            "internal filesystem capacity",
        )?;

        let agent_admission_capacity_bytes = multiply(
            input.concurrent_restores,
            input.maximum_agent_plan_bytes,
            "agent admission capacity",
        )?;
        let gateway_transport_window_bytes = multiply(
            input.gateway_instances,
            PRODUCTION_GATEWAY_IN_FLIGHT_BYTES,
            "gateway transport windows",
        )?;
        let admission_and_window_bytes = add(
            agent_admission_capacity_bytes,
            gateway_transport_window_bytes,
            "admission and gateway window capacity",
        )?;
        let required_source_pins = multiply(
            input.concurrent_restores,
            input.source_pins_per_restore,
            "source pin count",
        )?;
        let required_source_fds = multiply(
            input.concurrent_restores,
            input.source_fds_per_restore,
            "source descriptor count",
        )?;

        let physical_write_bytes_per_five_minutes = scale_ceil(
            input.peak_five_minute_unique_kv_bytes,
            input.physical_write_amplification_millis,
            PRODUCTION_CAPACITY_RATIO_MILLIS,
            "five-minute physical writes",
        )?;
        let physical_write_bytes_per_day = scale_ceil(
            input.daily_unique_kv_bytes,
            input.physical_write_amplification_millis,
            PRODUCTION_CAPACITY_RATIO_MILLIS,
            "daily physical writes",
        )?;
        let required_network_bits_per_second = scale_ceil(
            multiply(
                input.peak_network_payload_bytes_per_second,
                8,
                "network payload bits",
            )?,
            PRODUCTION_CAPACITY_RATIO_MILLIS,
            input.network_target_utilization_millis,
            "network line rate",
        )?;
        let backup_retention_bytes = multiply(
            input.backup_bytes_per_generation,
            input.retained_backup_generations,
            "backup retention",
        )?;

        Ok(Self {
            schema_version: CAPACITY_WORKSHEET_SCHEMA_VERSION,
            input,
            retained_object_bytes,
            pinned_object_bytes: input.pinned_object_bytes,
            evictable_object_bytes,
            staging_quota_bytes,
            store_quota_bytes,
            quarantine_cap_bytes,
            minimum_object_filesystem_bytes,
            estimated_catalog_live_bytes,
            wal_reserve_bytes: input.wal_reserve_bytes,
            catalog_backup_temporary_bytes,
            minimum_internal_filesystem_bytes,
            agent_admission_capacity_bytes,
            gateway_transport_window_bytes,
            admission_and_window_bytes,
            maximum_active_grants: input.concurrent_restores,
            required_source_pins,
            required_source_fds,
            endurance_bytes_per_five_minutes: input.peak_five_minute_new_object_bytes,
            physical_write_bytes_per_five_minutes,
            physical_write_bytes_per_day,
            device_write_budget_bytes_per_day: input.device_write_budget_bytes_per_day,
            endurance_within_device_budget: physical_write_bytes_per_day
                <= input.device_write_budget_bytes_per_day,
            physical_write_amplification_within_v1_gate: input.physical_write_amplification_millis
                <= PRODUCTION_WRITE_AMPLIFICATION_LIMIT_MILLIS,
            required_network_bits_per_second,
            backup_retention_bytes,
        })
    }
}

fn validate(input: CapacityWorksheetInput) -> Result<(), CapacityPlanningError> {
    if input.retained_unique_kv_bytes == 0 {
        return Err(CapacityPlanningError::Invalid(
            "retained unique KV bytes must be nonzero",
        ));
    }
    if input.catalog_row_count == 0
        || input.catalog_bytes_per_row == 0
        || input.catalog_bytes_per_row > MAX_CAPACITY_CATALOG_BYTES_PER_ROW
        || input.catalog_fixed_bytes == 0
        || input.wal_reserve_bytes == 0
    {
        return Err(CapacityPlanningError::Invalid(
            "catalog rows, row size, fixed bytes, or WAL reserve are outside bounds",
        ));
    }
    if input.concurrent_uploads == 0
        || input.concurrent_uploads > MAX_CAPACITY_CONCURRENCY
        || input.maximum_upload_bytes == 0
    {
        return Err(CapacityPlanningError::Invalid(
            "upload concurrency or maximum upload bytes are outside bounds",
        ));
    }
    if input.concurrent_restores == 0
        || input.concurrent_restores > MAX_CAPACITY_CONCURRENCY
        || input.maximum_agent_plan_bytes == 0
        || input.source_pins_per_restore > MAX_CAPACITY_CONCURRENCY
        || input.source_fds_per_restore > input.source_pins_per_restore
    {
        return Err(CapacityPlanningError::Invalid(
            "restore concurrency, plan bytes, pins, or descriptors are outside bounds",
        ));
    }
    if input.gateway_instances == 0 || input.gateway_instances > MAX_CAPACITY_GATEWAY_INSTANCES {
        return Err(CapacityPlanningError::Invalid(
            "gateway instance count is outside bounds",
        ));
    }
    if input.daily_unique_kv_bytes == 0
        || input.peak_five_minute_unique_kv_bytes > input.daily_unique_kv_bytes
        || input.peak_five_minute_new_object_bytes == 0
        || input.physical_write_amplification_millis == 0
        || input.physical_write_amplification_millis > MAX_CAPACITY_WRITE_AMPLIFICATION_MILLIS
        || input.device_write_budget_bytes_per_day == 0
    {
        return Err(CapacityPlanningError::Invalid(
            "write rate, amplification, or device budget is outside bounds",
        ));
    }
    if input.peak_network_payload_bytes_per_second == 0
        || input.network_target_utilization_millis == 0
        || input.network_target_utilization_millis > MAX_CAPACITY_NETWORK_UTILIZATION_MILLIS
    {
        return Err(CapacityPlanningError::Invalid(
            "network payload rate or target utilization is outside bounds",
        ));
    }
    if input.backup_bytes_per_generation == 0
        || input.retained_backup_generations == 0
        || input.retained_backup_generations > MAX_CAPACITY_BACKUP_GENERATIONS
    {
        return Err(CapacityPlanningError::Invalid(
            "backup generation bytes or retention count are outside bounds",
        ));
    }
    Ok(())
}

fn add(left: u64, right: u64, label: &'static str) -> Result<u64, CapacityPlanningError> {
    left.checked_add(right)
        .ok_or(CapacityPlanningError::Overflow(label))
}

fn multiply(left: u64, right: u64, label: &'static str) -> Result<u64, CapacityPlanningError> {
    let value = (left as u128)
        .checked_mul(right as u128)
        .ok_or(CapacityPlanningError::Overflow(label))?;
    u64::try_from(value).map_err(|_| CapacityPlanningError::Overflow(label))
}

fn scale_ceil(
    value: u64,
    numerator: u64,
    denominator: u64,
    label: &'static str,
) -> Result<u64, CapacityPlanningError> {
    let scaled = (value as u128)
        .checked_mul(numerator as u128)
        .ok_or(CapacityPlanningError::Overflow(label))?;
    let result = scaled.div_ceil(denominator as u128);
    u64::try_from(result).map_err(|_| CapacityPlanningError::Overflow(label))
}

fn scale_floor(
    value: u64,
    numerator: u64,
    denominator: u64,
    label: &'static str,
) -> Result<u64, CapacityPlanningError> {
    let scaled = (value as u128)
        .checked_mul(numerator as u128)
        .ok_or(CapacityPlanningError::Overflow(label))?;
    u64::try_from(scaled / denominator as u128).map_err(|_| CapacityPlanningError::Overflow(label))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CapacityWorksheetInput {
        CapacityWorksheetInput {
            retained_unique_kv_bytes: 750,
            retained_object_overhead_bytes: 0,
            pinned_object_bytes: 250,
            catalog_row_count: 10,
            catalog_bytes_per_row: 10,
            catalog_fixed_bytes: 50,
            wal_reserve_bytes: 75,
            concurrent_uploads: 2,
            maximum_upload_bytes: 100,
            concurrent_restores: 3,
            maximum_agent_plan_bytes: 200,
            source_pins_per_restore: 4,
            source_fds_per_restore: 2,
            gateway_instances: 2,
            daily_unique_kv_bytes: 1_000,
            peak_five_minute_unique_kv_bytes: 100,
            peak_five_minute_new_object_bytes: 110,
            physical_write_amplification_millis: 1_300,
            device_write_budget_bytes_per_day: 1_300,
            peak_network_payload_bytes_per_second: 100,
            network_target_utilization_millis: 800,
            backup_bytes_per_generation: 50,
            retained_backup_generations: 3,
        }
    }

    #[test]
    fn production_worksheet_covers_every_capacity_domain_exactly() {
        let report = CapacityWorksheet::production_v1(sample()).unwrap();
        assert_eq!(report.schema_version, 1);
        assert_eq!(report.retained_object_bytes, 750);
        assert_eq!(report.pinned_object_bytes, 250);
        assert_eq!(report.evictable_object_bytes, 500);
        assert_eq!(report.staging_quota_bytes, 200);
        assert_eq!(report.store_quota_bytes, 1_000);
        assert_eq!(report.quarantine_cap_bytes, 10);
        assert_eq!(report.minimum_object_filesystem_bytes, 1_010);
        assert_eq!(report.estimated_catalog_live_bytes, 150);
        assert_eq!(report.catalog_backup_temporary_bytes, 150);
        assert_eq!(report.minimum_internal_filesystem_bytes, 500);
        assert_eq!(report.agent_admission_capacity_bytes, 600);
        assert_eq!(report.gateway_transport_window_bytes, 67_108_864);
        assert_eq!(report.admission_and_window_bytes, 67_109_464);
        assert_eq!(report.maximum_active_grants, 3);
        assert_eq!(report.required_source_pins, 12);
        assert_eq!(report.required_source_fds, 6);
        assert_eq!(report.endurance_bytes_per_five_minutes, 110);
        assert_eq!(report.physical_write_bytes_per_five_minutes, 130);
        assert_eq!(report.physical_write_bytes_per_day, 1_300);
        assert!(report.endurance_within_device_budget);
        assert!(report.physical_write_amplification_within_v1_gate);
        assert_eq!(report.required_network_bits_per_second, 1_000);
        assert_eq!(report.backup_retention_bytes, 150);
    }

    #[test]
    fn staging_can_set_the_store_quota_and_network_rounds_up() {
        let mut input = sample();
        input.maximum_upload_bytes = 250;
        input.peak_network_payload_bytes_per_second = 1;
        input.network_target_utilization_millis = 950;
        let report = CapacityWorksheet::production_v1(input).unwrap();
        assert_eq!(report.staging_quota_bytes, 500);
        assert_eq!(report.store_quota_bytes, 1_250);
        assert_eq!(report.quarantine_cap_bytes, 12);
        assert_eq!(report.minimum_object_filesystem_bytes, 1_262);
        assert_eq!(report.required_network_bits_per_second, 9);
    }

    #[test]
    fn worksheet_rejects_invalid_relationships_and_overflow() {
        let mut invalid = sample();
        invalid.source_fds_per_restore = invalid.source_pins_per_restore + 1;
        assert!(matches!(
            CapacityWorksheet::production_v1(invalid),
            Err(CapacityPlanningError::Invalid(_))
        ));

        let mut invalid = sample();
        invalid.pinned_object_bytes = 751;
        assert!(matches!(
            CapacityWorksheet::production_v1(invalid),
            Err(CapacityPlanningError::Invalid(_))
        ));

        let mut overflow = sample();
        overflow.retained_unique_kv_bytes = u64::MAX;
        overflow.retained_object_overhead_bytes = 1;
        assert_eq!(
            CapacityWorksheet::production_v1(overflow),
            Err(CapacityPlanningError::Overflow("retained object bytes"))
        );
    }

    #[test]
    fn independent_capacity_products_fail_instead_of_saturating() {
        let cases = [
            {
                let mut input = sample();
                input.retained_unique_kv_bytes = u64::MAX;
                (input, "retained-data store quota")
            },
            {
                let mut input = sample();
                input.catalog_row_count = u64::MAX;
                input.catalog_bytes_per_row = MAX_CAPACITY_CATALOG_BYTES_PER_ROW;
                (input, "catalog row estimate")
            },
            {
                let mut input = sample();
                input.concurrent_uploads = MAX_CAPACITY_CONCURRENCY;
                input.maximum_upload_bytes = u64::MAX;
                (input, "staging quota")
            },
            {
                let mut input = sample();
                input.concurrent_restores = MAX_CAPACITY_CONCURRENCY;
                input.maximum_agent_plan_bytes = u64::MAX;
                (input, "agent admission capacity")
            },
            {
                let mut input = sample();
                input.daily_unique_kv_bytes = u64::MAX;
                input.physical_write_amplification_millis = MAX_CAPACITY_WRITE_AMPLIFICATION_MILLIS;
                (input, "daily physical writes")
            },
            {
                let mut input = sample();
                input.peak_network_payload_bytes_per_second = u64::MAX;
                (input, "network payload bits")
            },
            {
                let mut input = sample();
                input.backup_bytes_per_generation = u64::MAX;
                input.retained_backup_generations = MAX_CAPACITY_BACKUP_GENERATIONS;
                (input, "backup retention")
            },
        ];
        for (input, label) in cases {
            assert_eq!(
                CapacityWorksheet::production_v1(input),
                Err(CapacityPlanningError::Overflow(label))
            );
        }
    }

    #[test]
    fn write_budget_and_release_gate_are_independent_verdicts() {
        let mut input = sample();
        input.physical_write_amplification_millis = 1_301;
        input.device_write_budget_bytes_per_day = 1_000;
        let report = CapacityWorksheet::production_v1(input).unwrap();
        assert!(!report.physical_write_amplification_within_v1_gate);
        assert!(!report.endurance_within_device_budget);
        assert_eq!(report.physical_write_bytes_per_day, 1_301);
    }
}
