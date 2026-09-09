# @security/no-unsafe-3des

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_11f742618aab193d`

3DES 不得使用 ECB 模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyLedgerChannel {
  exportSnapshot(rows: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|PKCS5');
    cipher.init({ key: 'legacy-ledger' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(rows.join('#')));
  }
}

export function dumpLedger(rows: Array<string>): Uint8Array {
  const channel: LegacyLedgerChannel = new LegacyLedgerChannel();
  return channel.exportSnapshot(rows);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ModernLedgerChannel {
  exportSnapshot(rows: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CBC|PKCS7');
    cipher.init({ key: 'modern-ledger' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(rows.join('#')));
  }
}

export function dumpLedger(rows: Array<string>): Uint8Array {
  const channel: ModernLedgerChannel = new ModernLedgerChannel();
  return channel.exportSnapshot(rows);
}
```

### Rationale

ledger 代码触发 @security/no-unsafe-3des：3DES 不得使用 ECB 模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_4b6bdd90126cd356`

3DES 不得使用 ECB 模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class RemoteTunnelBuilder {
  private peers: Array<string>;

  constructor(peers: Array<string>) {
    this.peers = peers;
  }

  establish(): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|NoPadding');
    cipher.init({ key: 'remote-tunnel' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(this.peers.join(',')));
  }
}

export function openLegacyVpn(peers: Array<string>): Uint8Array {
  const builder: RemoteTunnelBuilder = new RemoteTunnelBuilder(peers);
  return builder.establish();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class RemoteTunnelBuilder {
  private peers: Array<string>;

  constructor(peers: Array<string>) {
    this.peers = peers;
  }

  establish(): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CFB|PKCS7');
    cipher.init({ key: 'remote-tunnel-2025' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(this.peers.join(',')));
  }
}

export function openModernVpn(peers: Array<string>): Uint8Array {
  const builder: RemoteTunnelBuilder = new RemoteTunnelBuilder(peers);
  return builder.establish();
}
```

### Rationale

vpn 代码触发 @security/no-unsafe-3des：3DES 不得使用 ECB 模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_5fcfa6a067e2f3cf`

3DES 不得使用 ECB 模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ArchiveTunnelBuilder {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|PKCS7');
    cipher.init({ key: 'archive-sync-legacy' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openLegacyArchiveChannel(peers: Array<string>): Uint8Array {
  const builder: ArchiveTunnelBuilder = new ArchiveTunnelBuilder();
  builder.describe(peers);
  return builder.seal(peers);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ArchiveTunnelBuilderSecure {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CBC|PKCS7');
    cipher.init({ key: 'archive-sync-modern' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openSecureArchiveChannel(peers: Array<string>): Uint8Array {
  const builder: ArchiveTunnelBuilderSecure = new ArchiveTunnelBuilderSecure();
  builder.describe(peers);
  return builder.seal(peers);
}
```

### Rationale

archive 代码触发 @security/no-unsafe-3des：3DES 不得使用 ECB 模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_b56c1b88d50b7482`

3DES 不得使用 ECB 模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ChannelTunnelBuilder {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|PKCS7');
    cipher.init({ key: 'channel-peers-legacy' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openLegacyChannelChannel(peers: Array<string>): Uint8Array {
  const builder: ChannelTunnelBuilder = new ChannelTunnelBuilder();
  builder.describe(peers);
  return builder.seal(peers);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ChannelTunnelBuilderSecure {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CFB|PKCS7');
    cipher.init({ key: 'channel-peers-modern' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openSecureChannelChannel(peers: Array<string>): Uint8Array {
  const builder: ChannelTunnelBuilderSecure = new ChannelTunnelBuilderSecure();
  builder.describe(peers);
  return builder.seal(peers);
}
```

### Rationale

channel 代码触发 @security/no-unsafe-3des：3DES 不得使用 ECB 模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_d34073e88ea46f1a`

3DES 不得使用 ECB 模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyTripleDesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB');
    cipher.init({ key: '3des-legacy' });
    return cipher.update(payload);
  }

  audit(labels: Array<string>): string {
    return labels.join('-');
  }
}

export function encryptWithLegacy3Des(text: string): Uint8Array {
  const channel: LegacyTripleDesChannel = new LegacyTripleDesChannel();
  channel.audit(['legacy', text]);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(text));
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureTripleDesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CBC|PKCS7');
    cipher.init({ key: '3des-secure' });
    return cipher.update(payload);
  }

  audit(labels: Array<string>): string {
    return labels.sort().join('-');
  }
}

export function encryptWithSafe3Des(text: string): Uint8Array {
  const channel: SecureTripleDesChannel = new SecureTripleDesChannel();
  channel.audit(['alpha', 'beta']);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(text));
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-3des：3DES 不得使用 ECB 模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
