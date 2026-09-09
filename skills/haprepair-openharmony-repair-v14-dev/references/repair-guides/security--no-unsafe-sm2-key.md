# @security/no-unsafe-sm2-key

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_154e45bd5a625b45`

SM2 密钥生成需显式指定安全曲线参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakSm2KeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('SM2|MD5');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-sm2-weak-${pair?.publicKey ?? 'unknown'}`;
  }

  record(alias: string): void {
    console.info(`sm2-weak-${alias}`);
  }
}

export function issueWeakSm2Key(alias: string): string {
  const factory: WeakSm2KeyFactory = new WeakSm2KeyFactory();
  factory.record(alias);
  return factory.generate(alias);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureSm2KeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('SM2_256|SHA256');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-sm2-strong-${pair?.publicKey ?? 'unknown'}`;
  }

  record(alias: string): void {
    console.info(`sm2-strong-${alias}`);
  }
}

export function issueSafeSm2Key(alias: string): string {
  const factory: SecureSm2KeyFactory = new SecureSm2KeyFactory();
  factory.record(alias);
  return factory.generate(alias);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-sm2-key：SM2 密钥生成需显式指定安全曲线参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_26cfbbbbfa1a4721`

SM2 密钥生成需显式指定安全曲线参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2');
  }
}

export function issueLegacyLogsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: LogsKeyPlant = new LogsKeyPlant();
  return factory.fabricate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA256');
  }
}

export function issueSecureLogsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: LogsKeyPlantSecure = new LogsKeyPlantSecure();
  return factory.fabricate();
}
```

### Rationale

logs 代码触发 @security/no-unsafe-sm2-key：SM2 密钥生成需显式指定安全曲线参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_5462cff33ba64375`

SM2 密钥生成需显式指定安全曲线参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacySm2Key(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('SM2|SHA256');
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function secureSm2Key(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA256');
}
```

### Rationale

invoice 代码触发 @security/no-unsafe-sm2-key：SM2 密钥生成需显式指定安全曲线参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_d23447a3ba75a2ea`

SM2 密钥生成需显式指定安全曲线参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2|SHA256');
  }
}

export function issueLegacyResultsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ResultsKeyPlant = new ResultsKeyPlant();
  return factory.fabricate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA384');
  }
}

export function issueSecureResultsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ResultsKeyPlantSecure = new ResultsKeyPlantSecure();
  return factory.fabricate();
}
```

### Rationale

results 代码触发 @security/no-unsafe-sm2-key：SM2 密钥生成需显式指定安全曲线参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_d26e9ad8d5da7da6`

SM2 密钥生成需显式指定安全曲线参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ProfileKeyFactory {
  make(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2');
  }
}

export function buildLegacyProfileKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ProfileKeyFactory = new ProfileKeyFactory();
  return factory.make();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ProfileKeyFactory {
  make(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA384');
  }
}

export function buildSecureProfileKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ProfileKeyFactory = new ProfileKeyFactory();
  return factory.make();
}
```

### Rationale

profile 代码触发 @security/no-unsafe-sm2-key：SM2 密钥生成需显式指定安全曲线参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
