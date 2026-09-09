# @security/no-unsafe-dh

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_2fc2bd2d4c35e522`

DH 协商算法必须使用 2048 bit 以上安全参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceOnboardingExchange {
  perform(deviceId: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1024');
    agreement.init({ peer: deviceId });
    return `${deviceId}:${agreement}`;
  }
}

export function onboardLegacyDevice(deviceId: string): string {
  const exchange: DeviceOnboardingExchange = new DeviceOnboardingExchange();
  return exchange.perform(deviceId);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceOnboardingExchange {
  perform(deviceId: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp4096');
    agreement.init({ peer: deviceId });
    return `${deviceId}:${agreement}`;
  }
}

export function onboardSecureDevice(deviceId: string): string {
  const exchange: DeviceOnboardingExchange = new DeviceOnboardingExchange();
  return exchange.perform(deviceId);
}
```

### Rationale

onboarding 代码触发 @security/no-unsafe-dh：DH 协商算法必须使用 2048 bit 以上安全参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_3b2b5878c420f35e`

DH 协商算法必须使用 2048 bit 以上安全参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhExchange {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1536');
    agreement.init({ peer: 'bridge-peer' });
    return this.annotate('legacy', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateLegacyBridge(): string {
  const exchange: BridgeDhExchange = new BridgeDhExchange();
  return exchange.start();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhExchangeSecure {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp6144');
    agreement.init({ peer: 'bridge-peer' });
    return this.annotate('secure', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateSecureBridge(): string {
  const exchange: BridgeDhExchangeSecure = new BridgeDhExchangeSecure();
  return exchange.start();
}
```

### Rationale

bridge 代码触发 @security/no-unsafe-dh：DH 协商算法必须使用 2048 bit 以上安全参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_68fef7548a846b61`

DH 协商算法必须使用 2048 bit 以上安全参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ClusterDhExchange {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1024');
    agreement.init({ peer: 'cluster-peer' });
    return this.annotate('legacy', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateLegacyCluster(): string {
  const exchange: ClusterDhExchange = new ClusterDhExchange();
  return exchange.start();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ClusterDhExchangeSecure {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp4096');
    agreement.init({ peer: 'cluster-peer' });
    return this.annotate('secure', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateSecureCluster(): string {
  const exchange: ClusterDhExchangeSecure = new ClusterDhExchangeSecure();
  return exchange.start();
}
```

### Rationale

cluster 代码触发 @security/no-unsafe-dh：DH 协商算法必须使用 2048 bit 以上安全参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_8031e2265f52b092`

DH 协商算法必须使用 2048 bit 以上安全参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakDhHandshake {
  negotiate(peerInfo: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1536');
    agreement.init({ peer: peerInfo });
    return `dh-weak-${peerInfo}`;
  }

  captureMetrics(events: Array<string>): number {
    return events.map((item) => item.length).reduce((a, b) => a + b, 0);
  }
}

export function startWeakDh(peer: string): string {
  const handshake: WeakDhHandshake = new WeakDhHandshake();
  handshake.captureMetrics([peer, 'handshake']);
  return handshake.negotiate(peer);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDhHandshake {
  negotiate(peerInfo: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp3072');
    agreement.init({ peer: peerInfo });
    return `dh-ok-${peerInfo.length}`;
  }

  trackPeer(latencies: Array<number>): number {
    return latencies.reduce((acc, v) => acc + v, 0);
  }
}

export function startSecureDh(peer: string): string {
  const handshake: SecureDhHandshake = new SecureDhHandshake();
  handshake.trackPeer([1, 3, peer.length]);
  return handshake.negotiate(peer);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-dh：DH 协商算法必须使用 2048 bit 以上安全参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_86f595d159e5bfa3`

DH 协商算法必须使用 2048 bit 以上安全参数。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyChatKeyExchange {
  startHandshake(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1536');
    agreement.init({ peer: 'support' });
    return this.makeTranscript('legacy', agreement);
  }

  private makeTranscript(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function legacyChatHandshake(): string {
  const exchange: LegacyChatKeyExchange = new LegacyChatKeyExchange();
  return exchange.startHandshake();
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureChatKeyExchange {
  startHandshake(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp3072');
    agreement.init({ peer: 'support' });
    return this.makeTranscript('modern', agreement);
  }

  private makeTranscript(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function secureChatHandshake(): string {
  const exchange: SecureChatKeyExchange = new SecureChatKeyExchange();
  return exchange.startHandshake();
}
```

### Rationale

chat 代码触发 @security/no-unsafe-dh：DH 协商算法必须使用 2048 bit 以上安全参数。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
