use super::*;

impl LocalStore {
    pub fn resolve_prefix(
        &self,
        nodes: &[PrefixNode],
        semantic: &Id32,
        family: &Id32,
        candidate_bound: usize,
    ) -> Result<Option<PrefixHit>, StoreError> {
        Ok(self
            .resolve_prefix_candidates(nodes, semantic, family, candidate_bound)?
            .into_iter()
            .next())
    }

    pub fn resolve_prefix_candidates(
        &self,
        nodes: &[PrefixNode],
        semantic: &Id32,
        family: &Id32,
        candidate_bound: usize,
    ) -> Result<Vec<PrefixHit>, StoreError> {
        let requested_tokens = nodes.last().map_or(0, |node| node.token_count);
        self.resolve_prefix_candidates_for_cut(
            nodes,
            semantic,
            family,
            candidate_bound,
            requested_tokens,
            self.config.minimum_readable_key_epoch,
        )
    }

    pub(crate) fn resolve_prefix_candidates_for_cut(
        &self,
        nodes: &[PrefixNode],
        semantic: &Id32,
        family: &Id32,
        candidate_bound: usize,
        requested_tokens: u64,
        minimum_key_epoch: u64,
    ) -> Result<Vec<PrefixHit>, StoreError> {
        let started = Instant::now();
        let started_unix_ns = current_unix_ns();
        let context = TraceContext::new(ServiceComponent::Store).ok();
        let result = self.resolve_prefix_candidates_inner(
            nodes,
            semantic,
            family,
            candidate_bound,
            minimum_key_epoch,
        );
        let (lookup, span) = match &result {
            Ok(candidates) if candidates.is_empty() => (LookupResult::Miss, SpanOutcome::Miss),
            Ok(candidates)
                if candidates
                    .first()
                    .is_some_and(|candidate| candidate.token_count == requested_tokens) =>
            {
                (LookupResult::Exact, SpanOutcome::Ok)
            }
            Ok(_) => (LookupResult::Ancestor, SpanOutcome::Ok),
            Err(StoreError::Integrity(_) | StoreError::Authentication(_)) => {
                (LookupResult::IntegrityError, SpanOutcome::IntegrityError)
            }
            Err(_) => (LookupResult::Unavailable, SpanOutcome::Unavailable),
        };
        let _ = self.telemetry.record_lookup(lookup);
        let _ = self
            .telemetry
            .observe_latency(TracePhase::ExactLookup, started.elapsed());
        if lookup == LookupResult::Miss {
            let _ = self.telemetry.record_fallback(FallbackReason::NoExactCut);
        }
        let _ = self
            .telemetry
            .set_health(HealthComponent::Catalog, result.is_ok());
        if let Some(context) = context {
            let _ = self.telemetry.record_span(
                &context,
                None,
                TracePhase::ExactLookup,
                span,
                started_unix_ns,
                current_unix_ns().max(started_unix_ns),
            );
        }
        result
    }

    fn resolve_prefix_candidates_inner(
        &self,
        nodes: &[PrefixNode],
        semantic: &Id32,
        family: &Id32,
        candidate_bound: usize,
        minimum_key_epoch: u64,
    ) -> Result<Vec<PrefixHit>, StoreError> {
        if nodes.is_empty() || candidate_bound == 0 {
            return Ok(Vec::new());
        }
        let result_bound = candidate_bound.min(64);
        let candidates = &nodes[nodes.len().saturating_sub(900)..];
        let mut sql = String::from(
            "SELECT manifest_id,token_count FROM prefix_checkpoints WHERE tenant=?1 AND semantic_id=?2 AND family_id=?3 AND prefix_id IN (",
        );
        for index in 0..candidates.len() {
            if index != 0 {
                sql.push(',');
            }
            sql.push('?');
            sql.push_str(&(index + 4).to_string());
        }
        let minimum_parameter = candidates.len() + 4;
        let active_parameter = minimum_parameter + 1;
        sql.push_str(") AND EXISTS(SELECT 1 FROM manifests m JOIN locations l ON l.tenant=m.tenant AND l.object_kind='manifest' AND l.object_id=m.manifest_id AND l.tier='local' AND l.state='AVAILABLE' WHERE m.tenant=prefix_checkpoints.tenant AND m.manifest_id=prefix_checkpoints.manifest_id AND m.key_epoch>=?");
        sql.push_str(&minimum_parameter.to_string());
        sql.push_str(" AND m.key_epoch<=?");
        sql.push_str(&active_parameter.to_string());
        sql.push_str(") AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=prefix_checkpoints.tenant AND t.object_kind='manifest' AND t.object_id=prefix_checkpoints.manifest_id) ORDER BY token_count DESC LIMIT ");
        sql.push_str(&result_bound.to_string());
        let connection = self.lock_catalog()?;
        let mut statement = connection.prepare(&sql)?;
        let mut values: Vec<&dyn rusqlite::ToSql> = vec![&self.tenant_namespace, semantic, family];
        for node in candidates {
            values.push(&node.id);
        }
        values.push(&minimum_key_epoch);
        values.push(&self.config.key_epoch);
        let mut rows = statement.query(values.as_slice())?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: Vec<u8> = row.get(0)?;
            let manifest_id = vec_id(raw)?;
            let token_count: u64 = row.get(1)?;
            let requested = nodes.last().unwrap().token_count;
            result.push(PrefixHit {
                manifest_id,
                token_count,
                recompute_tokens: requested.saturating_sub(token_count),
            });
        }
        Ok(result)
    }
}

fn current_unix_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
