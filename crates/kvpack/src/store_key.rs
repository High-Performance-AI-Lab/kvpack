use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use kvpack_core::{Id32, KeySchedule};
use rustix::fs::{Mode, OFlags};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use crate::error::{io_error, StoreError};

const KEY_BYTES: usize = 32;
const FILE_MODE: u32 = 0o600;
const DIR_MODE: u32 = 0o700;

/// Provider-loaded tenant key material. It is neither `Clone` nor serializable,
/// and callers can request only domain-separated schedules.
pub struct StoreKey {
    stable_root: [u8; KEY_BYTES],
    epoch_roots: BTreeMap<u64, Zeroizing<[u8; KEY_BYTES]>>,
    derive_all_epochs: bool,
}

impl StoreKey {
    fn from_master(root: [u8; KEY_BYTES]) -> Result<Self, StoreError> {
        validate_root(&root)?;
        Ok(Self {
            stable_root: root,
            epoch_roots: BTreeMap::new(),
            derive_all_epochs: true,
        })
    }

    /// Build provider-neutral key material from one stable identity root and
    /// independently erasable epoch roots. Custom KMS adapters use this
    /// constructor; provider names and locations never enter derived bytes.
    pub fn from_epoch_roots(
        stable_root: [u8; KEY_BYTES],
        epoch_roots: impl IntoIterator<Item = (u64, [u8; KEY_BYTES])>,
    ) -> Result<Self, StoreError> {
        validate_root(&stable_root)?;
        let mut roots = BTreeMap::new();
        for (epoch, root) in epoch_roots {
            if epoch == 0 || roots.contains_key(&epoch) {
                return Err(StoreError::State("provider key epoch set is invalid"));
            }
            validate_root(&root)?;
            roots.insert(epoch, Zeroizing::new(root));
        }
        if roots.is_empty() {
            return Err(StoreError::State("provider returned no epoch roots"));
        }
        Ok(Self {
            stable_root,
            epoch_roots: roots,
            derive_all_epochs: false,
        })
    }

    pub(crate) fn schedule(&self, tenant: &Id32, epoch: u64) -> Result<KeySchedule, StoreError> {
        if self.derive_all_epochs {
            return Ok(KeySchedule::derive(&self.stable_root, tenant, epoch)?);
        }
        let epoch_root = self
            .epoch_roots
            .get(&epoch)
            .ok_or(StoreError::State("key epoch is not loaded"))?;
        Ok(KeySchedule::derive_with_epoch_root(
            &self.stable_root,
            epoch_root,
            tenant,
            epoch,
        )?)
    }

    pub(crate) fn supports_epoch(&self, epoch: u64) -> bool {
        self.derive_all_epochs || self.epoch_roots.contains_key(&epoch)
    }

    fn uses_independent_epoch_roots(&self) -> bool {
        !self.derive_all_epochs
    }

    pub fn fingerprint(&self) -> String {
        hex(&Sha256::digest(self.stable_root.as_slice()))
    }
}

impl Drop for StoreKey {
    fn drop(&mut self) {
        self.stable_root.zeroize();
    }
}

impl fmt::Debug for StoreKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StoreKey(sha256:{})", self.fingerprint())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_root(root: &[u8; KEY_BYTES]) -> Result<(), StoreError> {
    if root.iter().all(|byte| *byte == 0) {
        return Err(StoreError::State("store key is all zero"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyEpochWindow {
    pub minimum_readable: u64,
    pub active: u64,
}

impl KeyEpochWindow {
    pub fn validate(self) -> Result<Self, StoreError> {
        if self.minimum_readable == 0
            || self.active < self.minimum_readable
            || self.active.saturating_sub(self.minimum_readable) >= 64
        {
            return Err(StoreError::State("key epoch window is invalid"));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyProviderQualification {
    Production,
    DevelopmentOnly,
}

pub trait KeyProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn qualification(&self) -> KeyProviderQualification;
    fn load_tenant_keys(
        &self,
        tenant: &[u8],
        window: KeyEpochWindow,
    ) -> Result<StoreKey, StoreError>;
}

pub fn load_store_key_from_provider(
    provider: &dyn KeyProvider,
    tenant: &[u8],
    window: KeyEpochWindow,
    require_production: bool,
) -> Result<StoreKey, StoreError> {
    let window = window.validate()?;
    if tenant.is_empty() {
        return Err(StoreError::State("key-provider tenant must be nonempty"));
    }
    if require_production && provider.qualification() != KeyProviderQualification::Production {
        return Err(StoreError::State(
            "development key provider is ineligible for production readiness",
        ));
    }
    let key = provider.load_tenant_keys(tenant, window)?;
    if require_production && !key.uses_independent_epoch_roots() {
        return Err(StoreError::State(
            "production key provider must supply independent epoch roots",
        ));
    }
    if !(window.minimum_readable..=window.active).all(|epoch| key.supports_epoch(epoch)) {
        return Err(StoreError::State(
            "key provider did not load the complete readable epoch window",
        ));
    }
    Ok(key)
}

pub struct FileKeyProvider {
    path: PathBuf,
    allowed_root: PathBuf,
}

impl FileKeyProvider {
    pub fn new(path: PathBuf, allowed_root: PathBuf) -> Self {
        Self { path, allowed_root }
    }
}

impl KeyProvider for FileKeyProvider {
    fn name(&self) -> &'static str {
        "file"
    }

    fn qualification(&self) -> KeyProviderQualification {
        KeyProviderQualification::DevelopmentOnly
    }

    fn load_tenant_keys(
        &self,
        _tenant: &[u8],
        window: KeyEpochWindow,
    ) -> Result<StoreKey, StoreError> {
        window.validate()?;
        StoreKey::from_master(read_store_key(&self.path, &self.allowed_root)?)
    }
}

pub struct InMemoryKeyProvider {
    stable_root: Zeroizing<[u8; KEY_BYTES]>,
    epoch_roots: BTreeMap<u64, Zeroizing<[u8; KEY_BYTES]>>,
    derive_all_epochs: bool,
}

impl InMemoryKeyProvider {
    pub fn from_master(root: [u8; KEY_BYTES]) -> Result<Self, StoreError> {
        validate_root(&root)?;
        Ok(Self {
            stable_root: Zeroizing::new(root),
            epoch_roots: BTreeMap::new(),
            derive_all_epochs: true,
        })
    }

    pub fn from_epoch_roots(
        stable_root: [u8; KEY_BYTES],
        epoch_roots: impl IntoIterator<Item = (u64, [u8; KEY_BYTES])>,
    ) -> Result<Self, StoreError> {
        validate_root(&stable_root)?;
        let mut roots = BTreeMap::new();
        for (epoch, root) in epoch_roots {
            if epoch == 0 || roots.contains_key(&epoch) {
                return Err(StoreError::State("provider key epoch set is invalid"));
            }
            validate_root(&root)?;
            roots.insert(epoch, Zeroizing::new(root));
        }
        if roots.is_empty() {
            return Err(StoreError::State("provider returned no epoch roots"));
        }
        Ok(Self {
            stable_root: Zeroizing::new(stable_root),
            epoch_roots: roots,
            derive_all_epochs: false,
        })
    }
}

impl KeyProvider for InMemoryKeyProvider {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn qualification(&self) -> KeyProviderQualification {
        KeyProviderQualification::DevelopmentOnly
    }

    fn load_tenant_keys(
        &self,
        _tenant: &[u8],
        window: KeyEpochWindow,
    ) -> Result<StoreKey, StoreError> {
        let window = window.validate()?;
        if self.derive_all_epochs {
            return StoreKey::from_master(*self.stable_root);
        }
        let roots = (window.minimum_readable..=window.active)
            .map(|epoch| {
                self.epoch_roots
                    .get(&epoch)
                    .map(|root| (epoch, **root))
                    .ok_or(StoreError::State("key epoch is not loaded"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        StoreKey::from_epoch_roots(*self.stable_root, roots)
    }
}

pub struct MacOsKeychainProvider {
    service: String,
    account_prefix: String,
}

impl MacOsKeychainProvider {
    pub fn new(service: String, account_prefix: String) -> Result<Self, StoreError> {
        validate_provider_labels(&service, &account_prefix)?;
        Ok(Self {
            service,
            account_prefix,
        })
    }
}

impl KeyProvider for MacOsKeychainProvider {
    fn name(&self) -> &'static str {
        "macos-keychain"
    }

    fn qualification(&self) -> KeyProviderQualification {
        KeyProviderQualification::Production
    }

    fn load_tenant_keys(
        &self,
        tenant: &[u8],
        window: KeyEpochWindow,
    ) -> Result<StoreKey, StoreError> {
        let window = window.validate()?;
        #[cfg(target_os = "macos")]
        {
            let tenant = tenant_handle(tenant);
            let stable = macos_keychain_lookup(
                &self.service,
                &format!("{}:{tenant}:stable", self.account_prefix),
            )?;
            let mut epochs = Vec::new();
            for epoch in window.minimum_readable..=window.active {
                epochs.push((
                    epoch,
                    macos_keychain_lookup(
                        &self.service,
                        &format!("{}:{tenant}:epoch:{epoch}", self.account_prefix),
                    )?,
                ));
            }
            StoreKey::from_epoch_roots(stable, epochs)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (&self.service, &self.account_prefix, tenant, window);
            Err(StoreError::State(
                "macOS Keychain provider is unavailable on this platform",
            ))
        }
    }
}

pub struct LinuxOsKeyStoreProvider {
    service: String,
}

impl LinuxOsKeyStoreProvider {
    pub fn new(service: String) -> Result<Self, StoreError> {
        validate_provider_labels(&service, "linux")?;
        Ok(Self { service })
    }
}

impl KeyProvider for LinuxOsKeyStoreProvider {
    fn name(&self) -> &'static str {
        "linux-os-key-store"
    }

    fn qualification(&self) -> KeyProviderQualification {
        KeyProviderQualification::Production
    }

    fn load_tenant_keys(
        &self,
        tenant: &[u8],
        window: KeyEpochWindow,
    ) -> Result<StoreKey, StoreError> {
        let window = window.validate()?;
        #[cfg(target_os = "linux")]
        {
            let tenant = tenant_handle(tenant);
            let stable = linux_key_store_lookup(&self.service, &tenant, "stable")?;
            let mut epochs = Vec::new();
            for epoch in window.minimum_readable..=window.active {
                epochs.push((
                    epoch,
                    linux_key_store_lookup(&self.service, &tenant, &format!("epoch:{epoch}"))?,
                ));
            }
            StoreKey::from_epoch_roots(stable, epochs)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (&self.service, tenant, window);
            Err(StoreError::State(
                "Linux OS key-store provider is unavailable on this platform",
            ))
        }
    }
}

fn validate_provider_labels(service: &str, account: &str) -> Result<(), StoreError> {
    if service.is_empty()
        || account.is_empty()
        || service.len() > 256
        || account.len() > 256
        || service.contains('\0')
        || account.contains('\0')
    {
        return Err(StoreError::State("key-provider label is invalid"));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn tenant_handle(tenant: &[u8]) -> String {
    hex(&Sha256::digest(tenant))
}

#[cfg(target_os = "macos")]
fn macos_keychain_lookup(service: &str, account: &str) -> Result<[u8; KEY_BYTES], StoreError> {
    let output = Command::new("/usr/bin/security")
        .args(["find-generic-password", "-w", "-s", service, "-a", account])
        .output()
        .map_err(io_error("invoke macOS Keychain"))?;
    decode_command_key(output, "macOS Keychain lookup failed")
}

#[cfg(target_os = "linux")]
fn linux_key_store_lookup(
    service: &str,
    tenant: &str,
    purpose: &str,
) -> Result<[u8; KEY_BYTES], StoreError> {
    let output = Command::new("secret-tool")
        .args([
            "lookup", "service", service, "tenant", tenant, "purpose", purpose,
        ])
        .output()
        .map_err(io_error("invoke Linux OS key store"))?;
    decode_command_key(output, "Linux OS key-store lookup failed")
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn decode_command_key(
    mut output: std::process::Output,
    failure: &'static str,
) -> Result<[u8; KEY_BYTES], StoreError> {
    if !output.status.success() {
        output.stdout.zeroize();
        output.stderr.zeroize();
        return Err(StoreError::State(failure));
    }
    let result = decode_hex_key(&output.stdout);
    output.stdout.zeroize();
    output.stderr.zeroize();
    result
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn decode_hex_key(bytes: &[u8]) -> Result<[u8; KEY_BYTES], StoreError> {
    let value = std::str::from_utf8(bytes)
        .map_err(|_| StoreError::State("OS key store returned a non-UTF-8 key"))?
        .trim();
    if value.len() != KEY_BYTES * 2 {
        return Err(StoreError::State(
            "OS key store must return exactly 64 hexadecimal characters",
        ));
    }
    let mut key = [0u8; KEY_BYTES];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| StoreError::State("OS key store returned invalid hexadecimal"))?;
    }
    if let Err(error) = validate_root(&key) {
        key.zeroize();
        return Err(error);
    }
    Ok(key)
}

fn resolved_parent(path: &Path, allowed_root: &Path, create: bool) -> Result<PathBuf, StoreError> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err(StoreError::State(
            "store-key path must be an absolute file path",
        ));
    }
    if create {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(DIR_MODE);
        builder
            .create(path.parent().unwrap())
            .map_err(io_error("create key directory"))?;
    }
    let root = allowed_root
        .canonicalize()
        .map_err(io_error("resolve key root"))?;
    let parent = path
        .parent()
        .unwrap()
        .canonicalize()
        .map_err(io_error("resolve key parent"))?;
    if parent != root && !parent.starts_with(&root) {
        return Err(StoreError::State("store-key path escapes allowed root"));
    }
    let metadata = parent
        .symlink_metadata()
        .map_err(io_error("inspect key parent"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(StoreError::State(
            "store-key parent is not a real directory",
        ));
    }
    fs::set_permissions(&parent, fs::Permissions::from_mode(DIR_MODE))
        .map_err(io_error("set key directory permissions"))?;
    Ok(parent)
}

pub fn create_store_key_random(path: &Path, allowed_root: &Path) -> Result<String, StoreError> {
    let parent = resolved_parent(path, allowed_root, true)?;
    if path.symlink_metadata().is_ok() {
        return Err(StoreError::State("store-key target already exists"));
    }
    let mut key = [0u8; KEY_BYTES];
    getrandom::fill(&mut key).map_err(|_| StoreError::State("store-key entropy failed"))?;
    let mut suffix = [0u8; 16];
    getrandom::fill(&mut suffix).map_err(|_| StoreError::State("store-key entropy failed"))?;
    let partial = parent.join(format!(".key-{}.partial", hex(&suffix)));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true).mode(FILE_MODE);
    let mut file = options
        .open(&partial)
        .map_err(io_error("create key partial"))?;
    file.write_all(&key)
        .map_err(io_error("write key partial"))?;
    file.sync_all().map_err(io_error("fsync key partial"))?;
    drop(file);
    fs::hard_link(&partial, path).map_err(io_error("publish key"))?;
    fs::remove_file(&partial).map_err(io_error("unlink key partial"))?;
    fs::File::open(&parent)
        .and_then(|dir| dir.sync_all())
        .map_err(io_error("fsync key directory"))?;
    let fingerprint = hex(&Sha256::digest(key));
    key.zeroize();
    Ok(fingerprint)
}

fn read_store_key(path: &Path, allowed_root: &Path) -> Result<[u8; KEY_BYTES], StoreError> {
    let parent = resolved_parent(path, allowed_root, false)?;
    let target = parent.join(path.file_name().unwrap());
    let fd = rustix::fs::open(
        &target,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(|errno| StoreError::Io {
        op: "open store key",
        source: std::io::Error::from(errno),
    })?;
    let mut file = fs::File::from(fd);
    let metadata = file.metadata().map_err(io_error("inspect store key"))?;
    if !metadata.is_file()
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
        || metadata.len() != KEY_BYTES as u64
    {
        return Err(StoreError::State(
            "store key must be private, single-link, and exactly 32 bytes",
        ));
    }
    let mut key = [0u8; KEY_BYTES];
    file.read_exact(&mut key)
        .map_err(io_error("read store key"))?;
    if let Err(error) = validate_root(&key) {
        key.zeroize();
        return Err(error);
    }
    Ok(key)
}

pub fn load_store_key(path: &Path, allowed_root: &Path) -> Result<StoreKey, StoreError> {
    StoreKey::from_master(read_store_key(path, allowed_root)?)
}
