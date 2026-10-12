use super::*;

impl LocalStore {
    pub(crate) fn reserve_restore_resources(
        &self,
        restore_id: Id32,
        resources: RestoreResourceCharge,
        limits: RestoreLimits,
    ) -> Result<(), StoreError> {
        let mut holds = self
            .restore_holds
            .lock()
            .map_err(|_| StoreError::State("restore hold mutex poisoned"))?;
        if holds.contains_key(&restore_id) {
            return Err(StoreError::State("restore hold identity collision"));
        }
        let aggregate =
            holds
                .values()
                .try_fold(RestoreResourceCharge::default(), |aggregate, hold| {
                    aggregate
                        .checked_add(hold.resources)
                        .ok_or(StoreError::Quota(
                            "concurrent restore resource accounting overflow",
                        ))
                })?;
        let aggregate = aggregate.checked_add(resources).ok_or(StoreError::Quota(
            "concurrent restore resource accounting overflow",
        ))?;
        if !charge_within_limits(aggregate, limits) {
            return Err(StoreError::Quota(
                "concurrent restore resources exceed declared limits",
            ));
        }
        holds.insert(
            restore_id,
            RestoreHold {
                pins: Vec::new(),
                resources,
            },
        );
        update_restore_gauges(&self.telemetry, &holds);
        Ok(())
    }

    pub(crate) fn attach_restore_pins(
        &self,
        restore_id: &Id32,
        pins: Vec<RetainedPin>,
    ) -> Result<(), StoreError> {
        let mut holds = self
            .restore_holds
            .lock()
            .map_err(|_| StoreError::State("restore hold mutex poisoned"))?;
        let hold = holds
            .get_mut(restore_id)
            .ok_or(StoreError::State("restore resource reservation is missing"))?;
        if !hold.pins.is_empty() {
            return Err(StoreError::State("restore source pins already attached"));
        }
        let pinned_bytes = pins.iter().try_fold(0u64, |total, pin| {
            total
                .checked_add(pin.bytes())
                .ok_or(StoreError::State("restore source pin byte total overflow"))
        })?;
        if pinned_bytes != hold.resources.pinned_source_bytes
            || pins.len() as u64 != hold.resources.source_pins
        {
            return Err(StoreError::Authentication(
                "restore source pins disagree with reserved resources",
            ));
        }
        hold.pins = pins;
        Ok(())
    }

    pub fn acknowledge_engine_free(&self, restore_id: &Id32) -> Result<bool, StoreError> {
        let mut holds = self
            .restore_holds
            .lock()
            .map_err(|_| StoreError::State("restore hold mutex poisoned"))?;
        let hold = holds.remove(restore_id);
        let found = hold.is_some();
        update_restore_gauges(&self.telemetry, &holds);
        drop(holds);
        if let Some(mut hold) = hold {
            // One catalog transaction for the complete hold instead of one
            // autocommit DELETE per pin.
            release_restore_pin_batch(self, &mut hold.pins, &[]);
        }
        Ok(found)
    }

    pub fn held_restore_count(&self) -> Result<usize, StoreError> {
        Ok(self
            .restore_holds
            .lock()
            .map_err(|_| StoreError::State("restore hold mutex poisoned"))?
            .len())
    }

    pub fn held_restore_resources(&self) -> Result<HeldRestoreResources, StoreError> {
        let holds = self
            .restore_holds
            .lock()
            .map_err(|_| StoreError::State("restore hold mutex poisoned"))?;
        let resources =
            holds
                .values()
                .try_fold(RestoreResourceCharge::default(), |aggregate, hold| {
                    aggregate
                        .checked_add(hold.resources)
                        .ok_or(StoreError::State("held restore resource total overflow"))
                })?;
        Ok(HeldRestoreResources {
            active_restores: holds.len() as u64,
            shadow_bytes: resources.shadow_bytes,
            pinned_source_bytes: resources.pinned_source_bytes,
            scratch_bytes: resources.scratch_bytes,
            staging_bytes: resources.staging_bytes,
            receive_window_bytes: resources.receive_window_bytes,
            safety_margin_bytes: resources.safety_margin_bytes,
            source_pins: resources.source_pins,
            source_fds: resources.source_fds,
        })
    }
}

fn update_restore_gauges(telemetry: &OperationalTelemetry, holds: &BTreeMap<Id32, RestoreHold>) {
    let aggregate = holds
        .values()
        .fold(RestoreResourceCharge::default(), |aggregate, hold| {
            aggregate
                .checked_add(hold.resources)
                .unwrap_or(RestoreResourceCharge {
                    shadow_bytes: u64::MAX,
                    pinned_source_bytes: u64::MAX,
                    scratch_bytes: u64::MAX,
                    staging_bytes: u64::MAX,
                    receive_window_bytes: u64::MAX,
                    safety_margin_bytes: u64::MAX,
                    source_pins: u64::MAX,
                    source_fds: u64::MAX,
                })
        });
    let in_flight = aggregate
        .shadow_bytes
        .saturating_add(aggregate.scratch_bytes)
        .saturating_add(aggregate.staging_bytes)
        .saturating_add(aggregate.receive_window_bytes);
    let _ = telemetry.set_resource(ResourceGauge::ActiveRestores, holds.len() as u64);
    let _ = telemetry.set_resource(ResourceGauge::InFlightBytes, in_flight);
    let _ = telemetry.set_resource(ResourceGauge::RestorePins, aggregate.source_pins);
    let _ = telemetry.set_resource(ResourceGauge::OpenDescriptors, aggregate.source_fds);
}
