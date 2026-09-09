# @security/no-unsafe-kdf

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_5755bb4bda0c310a`

PBKDF2/HKDF 派生不得使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakKdf {
  derive(password: string): string {
    const kdf = cryptoFramework.createKdf('PBKDF2|SHA1');
    kdf.init({ password, salt: 'legacy-salt' });
    return kdf.generate();
  }

  score(pieces: Array<string>): number {
    return pieces.map((p) => p.length).reduce((a, b) => a + b, 0);
  }
}

export function deriveWeakKey(password: string): string {
  const helper: WeakKdf = new WeakKdf();
  helper.score([password]);
  return helper.derive(password);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureKdf {
  derive(password: string): string {
    const kdf = cryptoFramework.createKdf('PBKDF2|SHA256');
    kdf.init({ password, salt: 'metrics-salt' });
    return kdf.generate();
  }

  score(pieces: Array<string>): number {
    return pieces.map((p) => p.length).reduce((a, b) => a + b, 0);
  }
}

export function deriveSecureKey(password: string): string {
  const helper: SecureKdf = new SecureKdf();
  helper.score([password, 'kdf']);
  return helper.derive(password);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-kdf：PBKDF2/HKDF 派生不得使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_7aa27b111f23a97c`

PBKDF2/HKDF 派生不得使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyArchiveKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('HKDF|SHA1');
  kdf.init({ text: seed });
  return kdf;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureArchiveKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('HKDF|SHA512');
  kdf.init({ text: seed });
  return kdf;
}
```

### Rationale

archive 代码触发 @security/no-unsafe-kdf：PBKDF2/HKDF 派生不得使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_85114a5184b7d77e`

PBKDF2/HKDF 派生不得使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyTokenKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('HKDF|SHA1');
  kdf.init({ text: seed });
  return kdf;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureTokenKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('HKDF|SHA512');
  kdf.init({ text: seed });
  return kdf;
}
```

### Rationale

token 代码触发 @security/no-unsafe-kdf：PBKDF2/HKDF 派生不得使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_d69093766d611ce4`

PBKDF2/HKDF 派生不得使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyProfileKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA1');
  kdf.init({ text: seed });
  return kdf;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureProfileKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA384');
  kdf.init({ text: seed });
  return kdf;
}
```

### Rationale

profile 代码触发 @security/no-unsafe-kdf：PBKDF2/HKDF 派生不得使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_daea36914c3025c7`

PBKDF2/HKDF 派生不得使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyBackupKey(secret: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA1');
  kdf.init({ text: secret });
  return kdf;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureBackupKey(secret: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA256');
  kdf.init({ text: secret });
  return kdf;
}
```

### Rationale

backup 代码触发 @security/no-unsafe-kdf：PBKDF2/HKDF 派生不得使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
