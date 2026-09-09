# Cryptographic Security Rules

Apply this specification to all `no-unsafe-*` cryptographic findings. Treat the
complete encrypt/decrypt, sign/verify, key-generation/import, storage, transport, and
migration path as one entity.

## Mandatory evidence

Identify the data format, algorithm identifier, key source and size, mode, padding,
IV/nonce, tag length, salt, iteration/work factor, persisted ciphertext, remote-peer
contract, HUKS alias/purpose, and backward-compatibility requirements. Find both
producer and consumer call sites. Never infer these facts from an example.

## Rule groups

- `no-unsafe-3des`, `no-unsafe-aes`, and `no-unsafe-sm4`: migrate algorithm/mode with
  authenticated-encryption semantics where supported. Preserve old-data access via
  an explicit versioned migration path; never reuse a nonce or silently reinterpret
  existing ciphertext.
- `no-unsafe-dh`, `no-unsafe-dh-key`, and `no-unsafe-ecdh`: validate groups/curves,
  key sizes, peer keys, shared-secret derivation, and KDF use. Coordinate both peers
  or implement version negotiation.
- `no-unsafe-dsa`, `no-unsafe-dsa-key`, `no-unsafe-ecdsa`,
  `no-unsafe-rsa-sign`, and `no-unsafe-sm2-key`: preserve signed payload encoding and
  verification interoperability while upgrading key/signature parameters.
- `no-unsafe-rsa-encrypt` and `no-unsafe-rsa-key`: preserve padding and key-format
  compatibility; use a hybrid construction for payload encryption when appropriate.
- `no-unsafe-hash` and `no-unsafe-mac`: distinguish checksums, password hashing,
  signatures, and message authentication. Upgrade every producer and verifier
  together; do not replace a digest name without a stored-data migration.
- `no-unsafe-kdf`: preserve password/secret encoding and stored parameter metadata;
  use per-record salts and a versioned work factor.
- `no-unsafe-huks`: keep alias, purpose, authorization, access control, and lifecycle
  consistent across generate/import/use/delete operations.
- `no-unsafe-sm2-cipher`: preserve SM2 ciphertext encoding/order and peer
  interoperability through explicit versioning or migration.

## Completion

Repair the full protocol path, add compatibility handling where existing data or
peers require it, and run available round-trip or known-answer tests. If the required
remote protocol or production key policy is genuinely external, record the exact
missing contract as `unresolved_external`; do not invent keys, salts, IVs, or test
vectors and do not count the alert as repaired.

