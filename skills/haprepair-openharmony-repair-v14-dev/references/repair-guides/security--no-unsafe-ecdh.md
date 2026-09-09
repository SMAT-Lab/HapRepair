# @security/no-unsafe-ecdh

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_101c6b8f8e4c195e`

ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class VehiclePairingKey {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC');
  }
}

export function legacyVehiclePairing(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: VehiclePairingKey = new VehiclePairingKey();
  return builder.build();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class VehiclePairingKey {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC256');
  }
}

export function secureVehiclePairing(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: VehiclePairingKey = new VehiclePairingKey();
  return builder.build();
}
```

### Rationale

vehicle 代码触发 @security/no-unsafe-ecdh：ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_11f580fc4df950f4`

ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakEcdhFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('ECC');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-ecdh-weak-${pair?.publicKey ?? 'unknown'}`;
  }

  monitor(peers: Array<string>): string {
    return peers.join('|');
  }
}

export function issueWeakEcdh(alias: string): string {
  const factory: WeakEcdhFactory = new WeakEcdhFactory();
  factory.monitor([alias, 'legacy']);
  return factory.generate(alias);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureEcdhFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('ECC256');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-ecdh-secure-${pair?.publicKey ?? 'unknown'}`;
  }

  monitor(peers: Array<string>): string {
    return peers.join('|');
  }
}

export function issueSafeEcdh(alias: string): string {
  const factory: SecureEcdhFactory = new SecureEcdhFactory();
  factory.monitor([alias]);
  return factory.generate(alias);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-ecdh：ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_36a57042b6046084`

ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SensorEcdhKeyBuilder {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC192');
  }
}

export function buildLegacySensorEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: SensorEcdhKeyBuilder = new SensorEcdhKeyBuilder();
  return builder.generate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SensorEcdhKeyBuilderSecure {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC384');
  }
}

export function buildSecureSensorEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: SensorEcdhKeyBuilderSecure = new SensorEcdhKeyBuilderSecure();
  return builder.generate();
}
```

### Rationale

sensor 代码触发 @security/no-unsafe-ecdh：ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_42047e32ab2bd488`

ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WearableKeyController {
  rotate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC192');
  }
}

export function legacyWearableKey(): cryptoFramework.AsymmetricKeyGenerator {
  const controller: WearableKeyController = new WearableKeyController();
  return controller.rotate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WearableKeyController {
  rotate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC384');
  }
}

export function secureWearableKey(): cryptoFramework.AsymmetricKeyGenerator {
  const controller: WearableKeyController = new WearableKeyController();
  return controller.rotate();
}
```

### Rationale

wearable 代码触发 @security/no-unsafe-ecdh：ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_c38c0445994132b2`

ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class GatewayEcdhKeyBuilder {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC');
  }
}

export function buildLegacyGatewayEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: GatewayEcdhKeyBuilder = new GatewayEcdhKeyBuilder();
  return builder.generate();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class GatewayEcdhKeyBuilderSecure {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC256');
  }
}

export function buildSecureGatewayEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: GatewayEcdhKeyBuilderSecure = new GatewayEcdhKeyBuilderSecure();
  return builder.generate();
}
```

### Rationale

gateway 代码触发 @security/no-unsafe-ecdh：ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
