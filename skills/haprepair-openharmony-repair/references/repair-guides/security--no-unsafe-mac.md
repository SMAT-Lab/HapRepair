# @security/no-unsafe-mac

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0f21e4dfe355db5e`

HMAC/MAC 算法禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyApiMac(body: string): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA1');
  mac.init({ text: body });
  return mac;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function secureApiMac(body: string): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA256');
  mac.init({ text: body });
  return mac;
}
```

### Rationale

api 代码触发 @security/no-unsafe-mac：HMAC/MAC 算法禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_13e4ab30dc4ec1b4`

HMAC/MAC 算法禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function createLegacyLicenseMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA1');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function createSecureLicenseMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA512');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
```

### Rationale

license 代码触发 @security/no-unsafe-mac：HMAC/MAC 算法禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_6e34a7541aa26359`

HMAC/MAC 算法禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakMac {
  compute(tag: string): string {
    const mac = cryptoFramework.createMac('SHA1');
    mac.init({ key: tag });
    mac.update(tag);
    return mac.doFinal();
  }

  aggregate(flags: Array<boolean>): number {
    return flags.filter((flag) => flag).length;
  }
}

export function buildWeakMac(tag: string): string {
  const helper: WeakMac = new WeakMac();
  helper.aggregate([false, true]);
  return helper.compute(tag);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureMac {
  compute(tag: string): string {
    const mac = cryptoFramework.createMac('SHA256');
    mac.init({ key: tag });
    mac.update(tag);
    return mac.doFinal();
  }

  aggregate(flags: Array<boolean>): number {
    return flags.filter((flag) => flag).length;
  }
}

export function buildSecureMac(tag: string): string {
  const helper: SecureMac = new SecureMac();
  helper.aggregate([true, false, tag.length > 2]);
  return helper.compute(tag);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-mac：HMAC/MAC 算法禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_9f3c9b75af9d0997`

HMAC/MAC 算法禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyWebhookMac(events: Array<string>): string {
  const mac = cryptoFramework.createMac('SHA1');
  events.forEach((event) => mac.update({ text: event }));
  return `${mac}`;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function secureWebhookMac(events: Array<string>): string {
  const mac = cryptoFramework.createMac('SHA512');
  events.forEach((event) => mac.update({ text: event }));
  return `${mac}`;
}
```

### Rationale

webhook 代码触发 @security/no-unsafe-mac：HMAC/MAC 算法禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_a31a24fca57ccb42`

HMAC/MAC 算法禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function createLegacyReportMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA1');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function createSecureReportMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA256');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
```

### Rationale

report 代码触发 @security/no-unsafe-mac：HMAC/MAC 算法禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
