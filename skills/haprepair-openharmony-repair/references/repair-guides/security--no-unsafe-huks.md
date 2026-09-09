# @security/no-unsafe-huks

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_46ecb705cbc6fe3c`

HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。

### Triggering pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

const keyAlias: string = 'weakKeyAlias';
const properties: Array<huks.HuksParam> = [
  { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_ECC },
  { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_ECC_KEY_SIZE_256 },
  { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_SIGN |
      huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_VERIFY },
  { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
  { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
  { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
];

const options: huks.HuksOptions = {
  properties: properties
};

export async function createWeakHuksKey(): Promise<void> {
  await huks.generateKeyItem(keyAlias, options);
}
```

### Repair pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

const simpleAlias: string = 'simple-key';
let simpleProperties: Array<huks.HuksParam> = [
  { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_ECC },
  { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_ECC_KEY_SIZE_256 },
  { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_SIGN | huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_VERIFY },
  { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_CBC },
  { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
  { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
];
const simpleOptions: huks.HuksOptions = { properties: simpleProperties };

huks.generateKeyItem(simpleAlias, simpleOptions);
```

### Rationale

simple 代码触发 @security/no-unsafe-huks：HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_7cc8e2f41aa4fe50`

HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。

### Triggering pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

class WeakHuksKey {
  async generate(alias: string): Promise<number> {
    const descriptor: string = 'ECB|SHA1|NONE';
    const options: huks.HuksOptions = this.buildOptions(descriptor);
    await huks.generateKeyItem(alias, options);
    return options.properties.length;
  }

  summarize(tags: Array<string>): string {
    return tags.join(':');
  }
  private buildOptions(descriptor: string): huks.HuksOptions {
    const lookup: Record<string, number> = {
      ECB: huks.HuksCipherMode.HUKS_MODE_ECB,
      SHA1: huks.HuksKeyDigest.HUKS_DIGEST_SHA1,
      NONE: huks.HuksKeyPadding.HUKS_PADDING_NONE
    };
    const [mode, digest, padding] = descriptor.split('|');
    const props: Array<huks.HuksParam> = [
        { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
        { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_128 },
        { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT },
        { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: lookup[mode] ?? huks.HuksCipherMode.HUKS_MODE_CBC },
        { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: lookup[digest] ?? huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
        { tag: huks.HuksTag.HUKS_TAG_PADDING, value: lookup[padding] ?? huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
    ];
    return { properties: props };
  }
}

export async function buildWeakHuksKey(alias: string): Promise<number> {
  const helper: WeakHuksKey = new WeakHuksKey();
  helper.summarize([alias, 'legacy']);
  return helper.generate(alias);
}
```

### Repair pattern

```arkts
import huks from '@ohos.security.huks';

class SecureHuksKey {
  async generate(alias: string): Promise<number> {
    const properties: Array<huks.HuksParam> = [
      { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
      { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_256 },
      { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT | huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_DECRYPT },
      { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_CBC },
      { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
      { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
    ];
    const options: huks.HuksOptions = { properties };
    await huks.generateKeyItem(alias, options);
    return properties.length;
  }

  summarize(tags: Array<string>): string {
    return tags.join(':');
  }
}

export async function buildSecureHuksKey(alias: string): Promise<number> {
  const helper: SecureHuksKey = new SecureHuksKey();
  helper.summarize([alias, 'secure']);
  return helper.generate(alias);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-huks：HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_87b4cca79b7b581f`

HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。

### Triggering pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

class LiteralWeakHuksKey {
  async generate(alias: string): Promise<number> {
    const options: huks.HuksOptions = {
      properties: [
        { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
        { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_128 },
        { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT },
        { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
        { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
        { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
      ]
    };
    await huks.generateKeyItem(alias, options);
    return options.properties.length;
  }

  describe(alias: string): string {
    return `${alias}-literal-weak`;
  }
}

export async function createLiteralWeakHuks(alias: string): Promise<number> {
  const factory: LiteralWeakHuksKey = new LiteralWeakHuksKey();
  factory.describe(alias);
  return factory.generate(alias);
}
```

### Repair pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

const literalAlias: string = 'literal-key';
const literalProperties: Array<huks.HuksParam> = [
  { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
  { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_256 },
  { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT | huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_DECRYPT },
  { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_CBC },
  { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
  { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
];
const literalOptions: huks.HuksOptions = { properties: literalProperties };

huks.generateKeyItem(literalAlias, literalOptions);
```

### Rationale

literal 代码触发 @security/no-unsafe-huks：HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_9ef3d67c8a65f908`

HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。

### Triggering pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

function createWalletKey(): void {
  let keyAlias: string = 'wallet-key';
  let properties: Array<huks.HuksParam> = [
    { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
    { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_128 },
    { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT },
    { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
    { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
    { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
  ];
  let options: huks.HuksOptions = { properties: properties };
  huks.generateKeyItem(keyAlias, options);
}

createWalletKey();
```

### Repair pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

function createWalletKey(): void {
  let keyAlias: string = 'wallet-key';
  let properties: Array<huks.HuksParam> = [
    { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
    { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_256 },
    { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT | huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_DECRYPT },
    { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_CBC },
    { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
    { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
  ];
  let options: huks.HuksOptions = { properties: properties };
  huks.generateKeyItem(keyAlias, options);
}

createWalletKey();
```

### Rationale

wallet 代码触发 @security/no-unsafe-huks：HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_cf0bdba09170fc6a`

HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。

### Triggering pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

function createAccessCardKey(): void {
  let keyAlias: string = 'access-card';
  let properties: Array<huks.HuksParam> = [
    { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_RSA },
    { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_RSA_KEY_SIZE_1024 },
    { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_SIGN },
    { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
    { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
    { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
  ];
  let options: huks.HuksOptions = { properties: properties };
  huks.generateKeyItem(keyAlias, options);
}

createAccessCardKey();
```

### Repair pattern

```arkts
import { huks } from '@kit.UniversalKeystoreKit';

function createAccessCardKey(): void {
  let keyAlias: string = 'access-card';
  let properties: Array<huks.HuksParam> = [
    { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_ECC },
    { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_ECC_KEY_SIZE_256 },
    { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_SIGN | huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_VERIFY },
    { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_CBC },
    { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
    { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
  ];
  let options: huks.HuksOptions = { properties: properties };
  huks.generateKeyItem(keyAlias, options);
}

createAccessCardKey();
```

### Rationale

access 代码触发 @security/no-unsafe-huks：HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
