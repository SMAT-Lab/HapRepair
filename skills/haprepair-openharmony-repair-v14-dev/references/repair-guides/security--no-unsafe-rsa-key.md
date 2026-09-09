# @security/no-unsafe-rsa-key

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0764a9e28406c356`

RSA 密钥长度必须不小于 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakRsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('RSA512|PRIMES_2');
    generator.init({ alias });
    return `${alias}-weak`;
  }

  history(): Array<string> {
    return ['legacy'];
  }
}

export function issueWeakRsaKey(alias: string): string {
  const factory: WeakRsaKeyFactory = new WeakRsaKeyFactory();
  factory.history();
  return factory.generate(alias);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureRsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('RSA3072|PRIMES_2');
    generator.init({ alias });
    return `${alias}-strong`;
  }

  history(): Array<string> {
    return ['init', 'rotate'];
  }
}

export function issueStrongRsaKey(alias: string): string {
  const factory: SecureRsaKeyFactory = new SecureRsaKeyFactory();
  factory.history();
  return factory.generate(alias);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-rsa-key：RSA 密钥长度必须不小于 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_12924968aefa5b87`

RSA 密钥长度必须不小于 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionRsaKeyPlant {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA1536|PRIMES_2');
  }
}

export function issueLegacySessionRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SessionRsaKeyPlant = new SessionRsaKeyPlant();
  return factory.produce();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionRsaKeyPlantSecure {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA4096|PRIMES_2');
  }
}

export function issueSecureSessionRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SessionRsaKeyPlantSecure = new SessionRsaKeyPlantSecure();
  return factory.produce();
}
```

### Rationale

session 代码触发 @security/no-unsafe-rsa-key：RSA 密钥长度必须不小于 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_3481f08ce8384560`

RSA 密钥长度必须不小于 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyLicenseKey(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('RSA1024|PRIMES_2');
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function modernLicenseKey(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('RSA4096|PRIMES_2');
}
```

### Rationale

license 代码触发 @security/no-unsafe-rsa-key：RSA 密钥长度必须不小于 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_ba3227f080071e48`

RSA 密钥长度必须不小于 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BatchKeyGenerator {
  allocate(): Array<cryptoFramework.AsymmetricKeyGenerator> {
    return ['user-a', 'user-b'].map(() => cryptoFramework.createAsyKeyGenerator('RSA1536|PRIMES_2'));
  }
}

export function createLegacyBatch(): Array<cryptoFramework.AsymmetricKeyGenerator> {
  const generator: BatchKeyGenerator = new BatchKeyGenerator();
  return generator.allocate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BatchKeyGenerator {
  allocate(): Array<cryptoFramework.AsymmetricKeyGenerator> {
    return ['user-a', 'user-b'].map(() => cryptoFramework.createAsyKeyGenerator('RSA3072|PRIMES_2'));
  }
}

export function createModernBatch(): Array<cryptoFramework.AsymmetricKeyGenerator> {
  const generator: BatchKeyGenerator = new BatchKeyGenerator();
  return generator.allocate();
}
```

### Rationale

batch 代码触发 @security/no-unsafe-rsa-key：RSA 密钥长度必须不小于 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_fe9019a24a838934`

RSA 密钥长度必须不小于 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretRsaKeyPlant {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA1024|PRIMES_2');
  }
}

export function issueLegacySecretRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SecretRsaKeyPlant = new SecretRsaKeyPlant();
  return factory.produce();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretRsaKeyPlantSecure {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA3072|PRIMES_2');
  }
}

export function issueSecureSecretRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SecretRsaKeyPlantSecure = new SecretRsaKeyPlantSecure();
  return factory.produce();
}
```

### Rationale

secret 代码触发 @security/no-unsafe-rsa-key：RSA 密钥长度必须不小于 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
