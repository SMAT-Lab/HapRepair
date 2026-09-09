# @security/no-unsafe-sm2-cipher

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1cfd54747957d6e5`

SM2 加解密禁止使用 MD5/SHA1 等弱摘要。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakSm2Cipher {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|MD5');
    cipher.init({ key: 'sm2-enc-weak' });
    return cipher.update(payload);
  }

  decrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|MD5');
    cipher.init({ key: 'sm2-enc-weak' });
    return cipher.update(payload);
  }
}

export function runSm2WeakCipher(text: string): Uint8Array {
  const cipher: WeakSm2Cipher = new WeakSm2Cipher();
  const encoder: TextEncoder = new TextEncoder();
  const encrypted: Uint8Array = cipher.encrypt(encoder.encode(text));
  return cipher.decrypt(encrypted);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureSm2Cipher {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
    cipher.init({ key: 'sm2-enc-strong' });
    return cipher.update(payload);
  }

  decrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
    cipher.init({ key: 'sm2-enc-strong' });
    return cipher.update(payload);
  }
}

export function runSm2SecureCipher(text: string): Uint8Array {
  const cipher: SecureSm2Cipher = new SecureSm2Cipher();
  const encoder: TextEncoder = new TextEncoder();
  const encrypted: Uint8Array = cipher.encrypt(encoder.encode(text));
  return cipher.decrypt(encrypted);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-sm2-cipher：SM2 加解密禁止使用 MD5/SHA1 等弱摘要。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_b5db8675801c7e5c`

SM2 加解密禁止使用 MD5/SHA1 等弱摘要。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyArchive(records: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA1');
  cipher.init({ text: records.join('|') });
  return cipher;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureArchive(records: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA512');
  cipher.init({ text: records.join('|') });
  return cipher;
}
```

### Rationale

archive 代码触发 @security/no-unsafe-sm2-cipher：SM2 加解密禁止使用 MD5/SHA1 等弱摘要。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_c326fd6d88a1c73d`

SM2 加解密禁止使用 MD5/SHA1 等弱摘要。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyTax(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA1');
  cipher.init({ text: payload });
  return cipher;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureTax(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
  cipher.init({ text: payload });
  return cipher;
}
```

### Rationale

tax 代码触发 @security/no-unsafe-sm2-cipher：SM2 加解密禁止使用 MD5/SHA1 等弱摘要。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_d5a1cc5889af1041`

SM2 加解密禁止使用 MD5/SHA1 等弱摘要。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsSm2Protector {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacyLogsSm2(payload: string): cryptoFramework.Cipher {
  const protector: LogsSm2Protector = new LogsSm2Protector();
  return protector.encrypt(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsSm2ProtectorSecure {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureLogsSm2(payload: string): cryptoFramework.Cipher {
  const protector: LogsSm2ProtectorSecure = new LogsSm2ProtectorSecure();
  return protector.encrypt(payload);
}
```

### Rationale

logs 代码触发 @security/no-unsafe-sm2-cipher：SM2 加解密禁止使用 MD5/SHA1 等弱摘要。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_e3e6f4fd99fa80f0`

SM2 加解密禁止使用 MD5/SHA1 等弱摘要。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsSm2Protector {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|MD5');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacyResultsSm2(payload: string): cryptoFramework.Cipher {
  const protector: ResultsSm2Protector = new ResultsSm2Protector();
  return protector.encrypt(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsSm2ProtectorSecure {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA512');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureResultsSm2(payload: string): cryptoFramework.Cipher {
  const protector: ResultsSm2ProtectorSecure = new ResultsSm2ProtectorSecure();
  return protector.encrypt(payload);
}
```

### Rationale

results 代码触发 @security/no-unsafe-sm2-cipher：SM2 加解密禁止使用 MD5/SHA1 等弱摘要。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
