# @security/no-unsafe-hash

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_27f86475898a2109`

禁止 MD5/SHA1 等弱哈希算法。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeLegacyBackupHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA1');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeSecureBackupHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA512');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
```

### Rationale

backup 代码触发 @security/no-unsafe-hash：禁止 MD5/SHA1 等弱哈希算法。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_4b1dbe72b570e7f9`

禁止 MD5/SHA1 等弱哈希算法。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function digestLegacyAttachment(payload: string): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('MD5');
  digest.update({ text: payload });
  return digest;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function digestSecureAttachment(payload: string): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA256');
  digest.update({ text: payload });
  return digest;
}
```

### Rationale

attachment 代码触发 @security/no-unsafe-hash：禁止 MD5/SHA1 等弱哈希算法。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_747d3adf1725a917`

禁止 MD5/SHA1 等弱哈希算法。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeLegacyManifestHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('MD5');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeSecureManifestHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA256');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
```

### Rationale

manifest 代码触发 @security/no-unsafe-hash：禁止 MD5/SHA1 等弱哈希算法。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_aead4c5875105e7e`

禁止 MD5/SHA1 等弱哈希算法。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakHasher {
  digest(data: string): string {
    const md = cryptoFramework.createMd('MD5');
    md.init();
    md.update(data);
    return md.digest();
  }

  collect(values: Array<number>): number {
    return values.reduce((acc, val) => acc * (val + 1), 1);
  }
}

export function hashWithMd5(message: string): string {
  const hasher: WeakHasher = new WeakHasher();
  hasher.collect([1, 2, message.length]);
  return hasher.digest(message);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureHasher {
  digest(data: string): string {
    const md = cryptoFramework.createMd('SHA256');
    md.init();
    md.update(data);
    return md.digest();
  }

  summarize(values: Array<number>): number {
    return values.reduce((acc, val) => acc + val, 0);
  }
}

export function hashSession(message: string): string {
  const hasher: SecureHasher = new SecureHasher();
  hasher.summarize([message.length, 10]);
  return hasher.digest(message);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-hash：禁止 MD5/SHA1 等弱哈希算法。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_c29034843a554a07`

禁止 MD5/SHA1 等弱哈希算法。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeLegacyChecksum(parts: Array<string>): string {
  const digest = cryptoFramework.createMd('SHA1');
  parts.forEach((part) => digest.update({ text: part }));
  return `${digest}`;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeSecureChecksum(parts: Array<string>): string {
  const digest = cryptoFramework.createMd('SHA512');
  parts.forEach((part) => digest.update({ text: part }));
  return `${digest}`;
}
```

### Rationale

checksum 代码触发 @security/no-unsafe-hash：禁止 MD5/SHA1 等弱哈希算法。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
