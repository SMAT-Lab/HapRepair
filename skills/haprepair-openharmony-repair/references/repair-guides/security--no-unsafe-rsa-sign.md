# @security/no-unsafe-rsa-sign

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1e0f3f4d805297b5`

RSA 签名需使用 PSS + SHA256/384 等安全散列。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyUpdateSignature(changelog: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA1536|PKCS1|MD5');
  signer.init({ text: changelog });
  return signer;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function modernUpdateSignature(changelog: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA4096|PSS|SHA384|MGF1_SHA384');
  signer.init({ text: changelog });
  return signer;
}
```

### Rationale

update 代码触发 @security/no-unsafe-rsa-sign：RSA 签名需使用 PSS + SHA256/384 等安全散列。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_42edf3d2a0e5ea27`

RSA 签名需使用 PSS + SHA256/384 等安全散列。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptRsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA1024|PKCS1|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyReceipt(payload: string): cryptoFramework.Signature {
  const signer: ReceiptRsaSigner = new ReceiptRsaSigner();
  return signer.execute(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptRsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureReceipt(payload: string): cryptoFramework.Signature {
  const signer: ReceiptRsaSignerSecure = new ReceiptRsaSignerSecure();
  return signer.execute(payload);
}
```

### Rationale

receipt 代码触发 @security/no-unsafe-rsa-sign：RSA 签名需使用 PSS + SHA256/384 等安全散列。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_ab00ab6a353f8cca`

RSA 签名需使用 PSS + SHA256/384 等安全散列。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakRsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('RSA1024|PKCS1|SHA1');
    signer.init({ key: 'rsa-legacy-sign' });
    signer.update(data);
    return signer.sign();
  }

  verify(data: string, sig: string): boolean {
    const verifier = cryptoFramework.createVerify('RSA1024|PKCS1|SHA1');
    verifier.init({ key: 'rsa-legacy-sign' });
    verifier.update(data);
    return verifier.verify(sig);
  }
}

export function runWeakRsaSign(message: string): boolean {
  const signer: WeakRsaSigner = new WeakRsaSigner();
  const sig: string = signer.sign(message);
  return signer.verify(message, sig);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureRsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
    signer.init({ key: 'rsa-strong-sign' });
    signer.update(data);
    return signer.sign();
  }

  verify(data: string, sig: string): boolean {
    const verifier = cryptoFramework.createVerify('RSA3072|PSS|SHA256|MGF1_SHA256');
    verifier.init({ key: 'rsa-strong-sign' });
    verifier.update(data);
    return verifier.verify(sig);
  }
}

export function runStrongRsaSign(message: string): boolean {
  const signer: SecureRsaSigner = new SecureRsaSigner();
  const sig: string = signer.sign(message);
  return signer.verify(message, sig);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-rsa-sign：RSA 签名需使用 PSS + SHA256/384 等安全散列。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_ad741c215ed82112`

RSA 签名需使用 PSS + SHA256/384 等安全散列。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class NoticeRsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA1536|PKCS1|MD5');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyNotice(payload: string): cryptoFramework.Signature {
  const signer: NoticeRsaSigner = new NoticeRsaSigner();
  return signer.execute(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class NoticeRsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA4096|PSS|SHA384|MGF1_SHA384');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureNotice(payload: string): cryptoFramework.Signature {
  const signer: NoticeRsaSignerSecure = new NoticeRsaSignerSecure();
  return signer.execute(payload);
}
```

### Rationale

notice 代码触发 @security/no-unsafe-rsa-sign：RSA 签名需使用 PSS + SHA256/384 等安全散列。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_dd8dae83df27bda4`

RSA 签名需使用 PSS + SHA256/384 等安全散列。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyContractSign(payload: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA1024|PKCS1|SHA1');
  signer.init({ text: payload });
  return signer;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

export function modernContractSign(payload: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
  signer.init({ text: payload });
  return signer;
}
```

### Rationale

contract 代码触发 @security/no-unsafe-rsa-sign：RSA 签名需使用 PSS + SHA256/384 等安全散列。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
