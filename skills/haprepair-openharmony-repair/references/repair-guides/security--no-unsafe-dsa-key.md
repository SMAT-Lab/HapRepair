# @security/no-unsafe-dsa-key

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0a817a14dd6b3310`

DSA 密钥模数需 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakDsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DSA1024');
    generator.init({ alias });
    return `${alias}-weak`;
  }

  notify(alias: string): void {
    console.info(`issued-${alias}`);
  }
}

export function issueWeakDsaKey(alias: string): string {
  const factory: WeakDsaKeyFactory = new WeakDsaKeyFactory();
  factory.notify(alias);
  return factory.generate(alias);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DSA3072');
    generator.init({ alias });
    return `key-created-${alias}`;
  }

  record(alias: string): string {
    return `${alias}-recorded`;
  }
}

export function issueStrongDsaKey(alias: string): string {
  const factory: SecureDsaKeyFactory = new SecureDsaKeyFactory();
  factory.record(alias);
  return factory.generate(alias);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-dsa-key：DSA 密钥模数需 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_0d28d2786a2c55a9`

DSA 密钥模数需 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceCertificateFactory {
  prepare(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA1536');
  }
}

export function legacyDeviceKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: DeviceCertificateFactory = new DeviceCertificateFactory();
  return factory.prepare();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceCertificateFactory {
  prepare(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function modernDeviceKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: DeviceCertificateFactory = new DeviceCertificateFactory();
  return factory.prepare();
}
```

### Rationale

device 代码触发 @security/no-unsafe-dsa-key：DSA 密钥模数需 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_45edf305bee0db23`

DSA 密钥模数需 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PatchSigningKeyFactory {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA1024');
  }
}

export function legacyPatchKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PatchSigningKeyFactory = new PatchSigningKeyFactory();
  return factory.build();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PatchSigningKeyFactory {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function modernPatchKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PatchSigningKeyFactory = new PatchSigningKeyFactory();
  return factory.build();
}
```

### Rationale

patch 代码触发 @security/no-unsafe-dsa-key：DSA 密钥模数需 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_7be772e056f43f53`

DSA 密钥模数需 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA1536');
  }
}

export function issueLegacyReceiptKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ReceiptKeyPlant = new ReceiptKeyPlant();
  return factory.fabricate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function issueSecureReceiptKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ReceiptKeyPlantSecure = new ReceiptKeyPlantSecure();
  return factory.fabricate();
}
```

### Rationale

receipt 代码触发 @security/no-unsafe-dsa-key：DSA 密钥模数需 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_b4f4e7d02f748ff6`

DSA 密钥模数需 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PolicyKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA2048');
  }
}

export function issueLegacyPolicyKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PolicyKeyPlant = new PolicyKeyPlant();
  return factory.fabricate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PolicyKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function issueSecurePolicyKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PolicyKeyPlantSecure = new PolicyKeyPlantSecure();
  return factory.fabricate();
}
```

### Rationale

policy 代码触发 @security/no-unsafe-dsa-key：DSA 密钥模数需 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
