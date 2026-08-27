use super::*;

impl LocalStore {
    pub(in crate::store) fn reconcile_work_dirs(&self, bound: usize) -> Result<(), StoreError> {
        self.reconcile_evictions(bound)?;
        self.reconcile_provisional_uploads(bound)?;
        let partials = self.config.object_root.join("partials");
        let quarantine = self.config.object_root.join("quarantine");
        for (index, entry) in fs::read_dir(&partials)
            .map_err(io_error("read partial directory"))?
            .enumerate()
        {
            if index >= bound {
                break;
            }
            let entry = entry.map_err(io_error("read partial entry"))?;
            if !entry
                .file_type()
                .map_err(io_error("inspect partial entry"))?
                .is_file()
            {
                continue;
            }
            let file_bytes = entry
                .metadata()
                .map_err(io_error("inspect restart partial"))?
                .len();
            let mut digest = Sha256::new();
            digest.update(b"kvpack/v1/restart-partial\0");
            digest.update(self.tenant_namespace);
            digest.update(entry.file_name().as_bytes());
            let entry_id: Id32 = digest.finalize().into();
            let path_token = format!("restart-{}.quarantine", hex(&entry_id));
            let created = now_ns();
            {
                let connection = self.lock_catalog()?;
                connection.execute(
                    "INSERT OR IGNORE INTO quarantine_entries(tenant,entry_id,object_kind,path_token,file_bytes,created_ns,expires_ns,reason) VALUES(?1,?2,'restart_partial',?3,?4,?5,?6,'partial recovered during restart')",
                    params![
                        self.tenant_namespace.as_slice(),
                        entry_id.as_slice(),
                        path_token,
                        file_bytes,
                        created,
                        created.saturating_add(24 * 60 * 60 * 1_000_000_000),
                    ],
                )?;
            }
            let destination = quarantine.join(&path_token);
            fs::rename(entry.path(), destination)
                .map_err(io_error("quarantine restart partial"))?;
        }
        for (index, entry) in fs::read_dir(&quarantine)
            .map_err(io_error("read quarantine directory"))?
            .enumerate()
        {
            if index >= bound {
                break;
            }
            let entry = entry.map_err(io_error("read quarantine entry"))?;
            if !entry
                .file_type()
                .map_err(io_error("inspect quarantine entry"))?
                .is_file()
            {
                continue;
            }
            let path_token = entry
                .file_name()
                .into_string()
                .map_err(|_| StoreError::State("quarantine filename is not valid UTF-8"))?;
            let connection = self.lock_catalog()?;
            let cataloged: u64 = connection.query_row(
                "SELECT COUNT(*) FROM quarantine_entries WHERE tenant=?1 AND path_token=?2",
                params![self.tenant_namespace.as_slice(), path_token],
                |row| row.get(0),
            )?;
            if cataloged == 0 {
                let mut digest = Sha256::new();
                digest.update(b"kvpack/v1/recovered-quarantine\0");
                digest.update(self.tenant_namespace);
                digest.update(path_token.as_bytes());
                let entry_id: Id32 = digest.finalize().into();
                let file_bytes = entry
                    .metadata()
                    .map_err(io_error("inspect recovered quarantine entry"))?
                    .len();
                let created = now_ns();
                connection.execute(
                    "INSERT OR IGNORE INTO quarantine_entries(tenant,entry_id,object_kind,path_token,file_bytes,created_ns,expires_ns,reason) VALUES(?1,?2,'recovered_quarantine',?3,?4,?5,?6,'uncataloged quarantine file recovered during restart')",
                    params![
                        self.tenant_namespace.as_slice(),
                        entry_id.as_slice(),
                        path_token,
                        file_bytes,
                        created,
                        created.saturating_add(24 * 60 * 60 * 1_000_000_000),
                    ],
                )?;
            }
        }
        let trash = self.config.object_root.join("trash");
        for (index, entry) in fs::read_dir(&trash)
            .map_err(io_error("read trash directory"))?
            .enumerate()
        {
            if index >= bound {
                break;
            }
            let entry = entry.map_err(io_error("read trash entry"))?;
            if entry
                .file_type()
                .map_err(io_error("inspect trash entry"))?
                .is_file()
            {
                fs::remove_file(entry.path()).map_err(io_error("finish trash unlink"))?;
            }
        }
        fsync_dir(&partials)?;
        fsync_dir(&quarantine)?;
        fsync_dir(&trash)?;
        Ok(())
    }
}
