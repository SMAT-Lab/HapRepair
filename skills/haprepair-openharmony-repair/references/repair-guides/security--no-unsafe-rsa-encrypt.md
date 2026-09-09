# @security/no-unsafe-rsa-encrypt

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_056953234e9e7da7`

RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionEnvelopeBuilder {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA1024|PKCS1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacySession(payload: string): cryptoFramework.Cipher {
  const builder: SessionEnvelopeBuilder = new SessionEnvelopeBuilder();
  return builder.seal(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionEnvelopeBuilderSecure {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA4096|PKCS1_OAEP|SHA256|MGF1_SHA256');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureSession(payload: string): cryptoFramework.Cipher {
  const builder: SessionEnvelopeBuilderSecure = new SessionEnvelopeBuilderSecure();
  return builder.seal(payload);
}
```

### Rationale

session 代码触发 @security/no-unsafe-rsa-encrypt：RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_4682ae1e77a10cff`

RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LicenseEnvelopeEncryptor {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA1024|PKCS1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function buildLegacyLicenseEnvelope(payload: string): cryptoFramework.Cipher {
  const encryptor: LicenseEnvelopeEncryptor = new LicenseEnvelopeEncryptor();
  return encryptor.encrypt(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LicenseEnvelopeEncryptor {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA3072|PKCS1_OAEP|SHA256|MGF1_SHA256');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function buildModernLicenseEnvelope(payload: string): cryptoFramework.Cipher {
  const encryptor: LicenseEnvelopeEncryptor = new LicenseEnvelopeEncryptor();
  return encryptor.encrypt(payload);
}
```

### Rationale

license 代码触发 @security/no-unsafe-rsa-encrypt：RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_65d446446968e68d`

RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyToken(token: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('RSA512|PKCS1');
  cipher.init({ text: token });
  return cipher;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureToken(token: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('RSA4096|PKCS1_OAEP|SHA256|MGF1_SHA256');
  cipher.init({ text: token });
  return cipher;
}
```

### Rationale

token 代码触发 @security/no-unsafe-rsa-encrypt：RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_ab041d98c776c058`

RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretEnvelopeBuilder {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA1536|PKCS1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacySecret(payload: string): cryptoFramework.Cipher {
  const builder: SecretEnvelopeBuilder = new SecretEnvelopeBuilder();
  return builder.seal(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretEnvelopeBuilderSecure {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA3072|PKCS1_OAEP|SHA384|MGF1_SHA384');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureSecret(payload: string): cryptoFramework.Cipher {
  const builder: SecretEnvelopeBuilderSecure = new SecretEnvelopeBuilderSecure();
  return builder.seal(payload);
}
```

### Rationale

secret 代码触发 @security/no-unsafe-rsa-encrypt：RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_c814877a9c40853a`

RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakRsaEncryptor {
  encrypt(data: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('RSA512|PKCS1');
    cipher.init({ key: 'rsa-legacy' });
    return cipher.update(data);
  }

  queue(messages: Array<string>): number {
    return messages.length * 2;
  }
}

export function encryptWithWeakRsa(text: string): Uint8Array {
  const encryptor: WeakRsaEncryptor = new WeakRsaEncryptor();
  encryptor.queue([text, 'legacy']);
  const encoder: TextEncoder = new TextEncoder();
  return encryptor.encrypt(encoder.encode(text));
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureRsaEncryptor {
  encrypt(data: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('RSA3072|PKCS1_OAEP|SHA256|MGF1_SHA256');
    cipher.init({ key: 'rsa-strong-key' });
    return cipher.update(data);
  }

  queue(messages: Array<string>): number {
    return messages.length;
  }
}

export function encryptWithStrongRsa(text: string): Uint8Array {
  const encryptor: SecureRsaEncryptor = new SecureRsaEncryptor();
  encryptor.queue([text, text.slice(0, 2)]);
  const encoder: TextEncoder = new TextEncoder();
  return encryptor.encrypt(encoder.encode(text));
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-rsa-encrypt：RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
