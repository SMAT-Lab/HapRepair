# @security/no-unsafe-aes

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_54f17248da6baea3`

AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ResearchArchiveChannel {
  private departmentTag: string;

  constructor(tag: string) {
    this.departmentTag = tag;
  }

  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|ECB|NoPadding');
    cipher.init({ key: `${this.departmentTag}-legacy-key` });
    return cipher.update(payload);
  }

  aggregateMarks(values: Array<number>): number {
    return values.reduce((acc, current) => acc + current, 0);
  }
}

export function archiveResearch(findings: Array<string>): Uint8Array {
  const encoder: TextEncoder = new TextEncoder();
  const channel: ResearchArchiveChannel = new ResearchArchiveChannel('bio-lab');
  const payload: Uint8Array = encoder.encode(findings.join('|'));
  channel.aggregateMarks([findings.length, payload.length]);
  return channel.encrypt(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ResearchArchiveChannel {
  private departmentTag: string;

  constructor(tag: string) {
    this.departmentTag = tag;
  }

  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|GCM|PKCS7');
    cipher.init({ key: `${this.departmentTag}-gcm-key` });
    return cipher.update(payload);
  }

  aggregateMarks(values: Array<number>): number {
    return values.reduce((acc, current) => acc + current, 0);
  }
}

export function archiveResearch(findings: Array<string>): Uint8Array {
  const encoder: TextEncoder = new TextEncoder();
  const channel: ResearchArchiveChannel = new ResearchArchiveChannel('bio-lab');
  const payload: Uint8Array = encoder.encode(findings.join('|'));
  channel.aggregateMarks([findings.length, payload.length]);
  return channel.encrypt(payload);
}
```

### Rationale

archive 代码触发 @security/no-unsafe-aes：AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_7f0372073cf9812b`

AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionTicketAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|ECB|NoPadding');
    cipher.init({ key: 'session-ticket-legacy' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapLegacySessionTicket(records: Array<string>): Uint8Array {
  const channel: SessionTicketAesChannel = new SessionTicketAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionTicketAesChannelSecure {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|CBC|PKCS7');
    cipher.init({ key: 'session-ticket-secure' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapSecureSessionTicket(records: Array<string>): Uint8Array {
  const channel: SessionTicketAesChannelSecure = new SessionTicketAesChannelSecure();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
```

### Rationale

session 代码触发 @security/no-unsafe-aes：AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_8715ffe5b997a8ce`

AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class VideoStreamProtector {
  private frames: Array<Uint8Array> = [];

  constructor(streamId: string) {
    const encoder: TextEncoder = new TextEncoder();
    this.frames.push(encoder.encode(streamId));
  }

  cacheFrame(text: string): void {
    const encoder: TextEncoder = new TextEncoder();
    this.frames.push(encoder.encode(text));
  }

  sealChannel(): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|ECB|PKCS5');
    cipher.init({ key: 'stream-legacy-key' });
    return cipher.update(this.combineFrames());
  }

  private combineFrames(): Uint8Array {
    const totalLength: number = this.frames.reduce((acc, frame) => acc + frame.length, 0);
    const merged: Uint8Array = new Uint8Array(totalLength);
    let offset: number = 0;
    this.frames.forEach((frame) => {
      merged.set(frame, offset);
      offset += frame.length;
    });
    return merged;
  }
}

export function buildLegacyVideoBundle(frames: Array<string>): Uint8Array {
  const protector: VideoStreamProtector = new VideoStreamProtector('lecture-channel');
  frames.forEach((frame) => protector.cacheFrame(frame));
  return protector.sealChannel();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class VideoStreamProtector {
  private frames: Array<Uint8Array> = [];

  constructor(streamId: string) {
    const encoder: TextEncoder = new TextEncoder();
    this.frames.push(encoder.encode(streamId));
  }

  cacheFrame(text: string): void {
    const encoder: TextEncoder = new TextEncoder();
    this.frames.push(encoder.encode(text));
  }

  sealChannel(): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|GCM|PKCS7');
    cipher.init({ key: 'stream-modern-key' });
    return cipher.update(this.combineFrames());
  }

  private combineFrames(): Uint8Array {
    const totalLength: number = this.frames.reduce((acc, frame) => acc + frame.length, 0);
    const merged: Uint8Array = new Uint8Array(totalLength);
    let offset: number = 0;
    this.frames.forEach((frame) => {
      merged.set(frame, offset);
      offset += frame.length;
    });
    return merged;
  }
}

export function buildSecureVideoBundle(frames: Array<string>): Uint8Array {
  const protector: VideoStreamProtector = new VideoStreamProtector('lecture-channel');
  frames.forEach((frame) => protector.cacheFrame(frame));
  return protector.sealChannel();
}
```

### Rationale

stream 代码触发 @security/no-unsafe-aes：AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_897ea65909fa6852`

AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class InsecureAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES128|ECB|NoPadding');
    cipher.init({ key: 'legacy-key-128' });
    return cipher.update(payload);
  }

  checksum(values: Array<number>): number {
    return values.reduce((acc, val) => acc ^ val, 0);
  }
}

export function encryptWithLegacyMode(message: string): Uint8Array {
  const channel: InsecureAesChannel = new InsecureAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(message);
  channel.checksum([message.length, 42]);
  return channel.encrypt(body);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES128|GCM|PKCS5');
    cipher.init({ key: 'analytics-key-256' });
    return cipher.update(payload);
  }

  summarize(readings: Array<number>): number {
    return readings.reduce((acc, v) => acc + v, 0);
  }
}

export function protectTelemetry(message: string): Uint8Array {
  const channel: SecureAesChannel = new SecureAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(message);
  channel.summarize([1, 2, 3, message.length]);
  return channel.encrypt(body);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-aes：AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_fd6e65668fe73d70`

AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupVaultAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|ECB|PKCS5');
    cipher.init({ key: 'backup-vault-legacy' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapLegacyBackupVault(records: Array<string>): Uint8Array {
  const channel: BackupVaultAesChannel = new BackupVaultAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupVaultAesChannelSecure {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|GCM|PKCS7');
    cipher.init({ key: 'backup-vault-secure' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapSecureBackupVault(records: Array<string>): Uint8Array {
  const channel: BackupVaultAesChannelSecure = new BackupVaultAesChannelSecure();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
```

### Rationale

backup 代码触发 @security/no-unsafe-aes：AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
