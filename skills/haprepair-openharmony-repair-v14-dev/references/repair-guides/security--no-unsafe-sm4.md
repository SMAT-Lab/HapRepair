# @security/no-unsafe-sm4

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_094bacc3973da2e5`

SM4 禁止使用 ECB 等不安全分组模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyPassport(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS7');
  cipher.init({ text: payload });
  return cipher;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecurePassport(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|CBC|PKCS7');
  cipher.init({ text: payload });
  return cipher;
}
```

### Rationale

passport 代码触发 @security/no-unsafe-sm4：SM4 禁止使用 ECB 等不安全分组模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_160d7a27faa15fff`

SM4 禁止使用 ECB 等不安全分组模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakSm4Channel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM4|ECB|PKCS7');
    cipher.init({ key: 'sm4-legacy' });
    return cipher.update(payload);
  }

  digestFrames(frames: Array<number>): number {
    return frames.reduce((acc, val) => acc + val, 0);
  }
}

export function encryptWithWeakSm4(message: string): Uint8Array {
  const channel: WeakSm4Channel = new WeakSm4Channel();
  channel.digestFrames([message.length]);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(message));
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureSm4Channel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM4|CBC|PKCS7');
    cipher.init({ key: 'sm4-secure' });
    return cipher.update(payload);
  }

  digestFrames(frames: Array<number>): number {
    return frames.reduce((acc, val) => acc + val, 0);
  }
}

export function encryptWithSafeSm4(message: string): Uint8Array {
  const channel: SecureSm4Channel = new SecureSm4Channel();
  channel.digestFrames([message.length, 10]);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(message));
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-sm4：SM4 禁止使用 ECB 等不安全分组模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_24ad63cd4a82045e`

SM4 禁止使用 ECB 等不安全分组模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class TicketsSm4Channel {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptLegacyTicketsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: TicketsSm4Channel = new TicketsSm4Channel();
  return channel.seal(records);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class TicketsSm4ChannelSecure {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|CBC|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptSecureTicketsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: TicketsSm4ChannelSecure = new TicketsSm4ChannelSecure();
  return channel.seal(records);
}
```

### Rationale

tickets 代码触发 @security/no-unsafe-sm4：SM4 禁止使用 ECB 等不安全分组模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_51c33d1b82af121d`

SM4 禁止使用 ECB 等不安全分组模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordsSm4Channel {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptLegacyRecordsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: RecordsSm4Channel = new RecordsSm4Channel();
  return channel.seal(records);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordsSm4ChannelSecure {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|CFB|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptSecureRecordsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: RecordsSm4ChannelSecure = new RecordsSm4ChannelSecure();
  return channel.seal(records);
}
```

### Rationale

records 代码触发 @security/no-unsafe-sm4：SM4 禁止使用 ECB 等不安全分组模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_f26a9b9c441ec623`

SM4 禁止使用 ECB 等不安全分组模式。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyLogs(logs: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS5');
  cipher.init({ text: logs.join(';') });
  return cipher;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureLogs(logs: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|CFB|PKCS7');
  cipher.init({ text: logs.join(';') });
  return cipher;
}
```

### Rationale

logs 代码触发 @security/no-unsafe-sm4：SM4 禁止使用 ECB 等不安全分组模式。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
