# @security/no-unsafe-dh-key

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_349d7c1cee0358d0`

DH 密钥模数必须 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class TrainingKeyFactory {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp1024');
  }
}

export function legacyTrainingKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: TrainingKeyFactory = new TrainingKeyFactory();
  return factory.generate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class TrainingKeyFactory {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp4096');
  }
}

export function secureTrainingKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: TrainingKeyFactory = new TrainingKeyFactory();
  return factory.generate();
}
```

### Rationale

training 代码触发 @security/no-unsafe-dh-key：DH 密钥模数必须 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_816401950e508ca8`

DH 密钥模数必须 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhKeyFactory {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp1536');
  }
}

export function buildLegacyBridgeDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BridgeDhKeyFactory = new BridgeDhKeyFactory();
  return factory.create();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhKeyFactorySecure {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp6144');
  }
}

export function buildSecureBridgeDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BridgeDhKeyFactorySecure = new BridgeDhKeyFactorySecure();
  return factory.create();
}
```

### Rationale

bridge 代码触发 @security/no-unsafe-dh-key：DH 密钥模数必须 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_a2087ab5ec34b14a`

DH 密钥模数必须 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SyncKeyMaterial {
  refresh(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp1536');
  }
}

export function requestLegacyDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const material: SyncKeyMaterial = new SyncKeyMaterial();
  return material.refresh();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SyncKeyMaterial {
  refresh(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp6144');
  }
}

export function requestModernDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const material: SyncKeyMaterial = new SyncKeyMaterial();
  return material.refresh();
}
```

### Rationale

sync 代码触发 @security/no-unsafe-dh-key：DH 密钥模数必须 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_e4ff06db8212776c`

DH 密钥模数必须 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyDhKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DH_modp1024');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-legacy-${pair?.publicKey ?? 'unknown'}`;
  }

  collectAuditTrail(updated: Array<string>): number {
    return updated.length;
  }
}

export function createWeakDhKey(alias: string): string {
  const factory: LegacyDhKeyFactory = new LegacyDhKeyFactory();
  factory.collectAuditTrail([alias, 'rotate']);
  return factory.generate(alias);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDhKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DH_modp3072');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-secure-${pair?.publicKey ?? 'unknown'}`;
  }

  rotateHistory(oldKeys: Array<string>): string {
    return oldKeys.join('|');
  }
}

export function createStrongDhKey(alias: string): string {
  const factory: SecureDhKeyFactory = new SecureDhKeyFactory();
  factory.rotateHistory([alias, 'previous']);
  return factory.generate(alias);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-dh-key：DH 密钥模数必须 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_f56d057cc35b64e5`

DH 密钥模数必须 >= 2048 bit。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupDhKeyFactory {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp2048');
  }
}

export function buildLegacyBackupDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BackupDhKeyFactory = new BackupDhKeyFactory();
  return factory.create();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupDhKeyFactorySecure {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp8192');
  }
}

export function buildSecureBackupDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BackupDhKeyFactorySecure = new BackupDhKeyFactorySecure();
  return factory.create();
}
```

### Rationale

backup 代码触发 @security/no-unsafe-dh-key：DH 密钥模数必须 >= 2048 bit。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
