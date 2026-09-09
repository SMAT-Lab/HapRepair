# @security/no-unsafe-ecdsa

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_4dee7b3d1478ef63`

ECDSA 签名/验签禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakEcdsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('ECC224|SHA1');
    signer.init({ key: 'ecc-legacy-key' });
    signer.update(data);
    return signer.sign();
  }

  verify(data: string, sig: string): boolean {
    const verifier = cryptoFramework.createVerify('ECC224|SHA1');
    verifier.init({ key: 'ecc-legacy-key' });
    verifier.update(data);
    return verifier.verify(sig);
  }
}

export function signWithWeakEcdsa(message: string): boolean {
  const handler: WeakEcdsaSigner = new WeakEcdsaSigner();
  const sig: string = handler.sign(message);
  return handler.verify(message, sig);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureEcdsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('ECC256|SHA256');
    signer.init({ key: 'ecc-safe-key' });
    signer.update(data);
    return signer.sign();
  }

  verifySignature(data: string, sig: string): boolean {
    const verifier = cryptoFramework.createVerify('ECC256|SHA256');
    verifier.init({ key: 'ecc-safe-key' });
    verifier.update(data);
    return verifier.verify(sig);
  }
}

export function runSecureEcdsa(message: string): boolean {
  const handler: SecureEcdsaSigner = new SecureEcdsaSigner();
  const sig: string = handler.sign(message);
  return handler.verifySignature(message, sig);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-ecdsa：ECDSA 签名/验签禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_553c49eabc552fc5`

ECDSA 签名/验签禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PassEcdsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC224|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyPassEcdsa(payload: string): cryptoFramework.Signature {
  const signer: PassEcdsaSigner = new PassEcdsaSigner();
  return signer.execute(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PassEcdsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC384|SHA256');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecurePassEcdsa(payload: string): cryptoFramework.Signature {
  const signer: PassEcdsaSignerSecure = new PassEcdsaSignerSecure();
  return signer.execute(payload);
}
```

### Rationale

pass 代码触发 @security/no-unsafe-ecdsa：ECDSA 签名/验签禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_a04aed13cdfad08e`

ECDSA 签名/验签禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class TrustBadgeSigner {
  sign(statement: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC224|SHA1');
    signer.init({ text: statement });
    return signer;
  }
}

export function signLegacyBadge(statement: string): cryptoFramework.Signature {
  const signer: TrustBadgeSigner = new TrustBadgeSigner();
  return signer.sign(statement);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class TrustBadgeSigner {
  sign(statement: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC256|SHA256');
    signer.init({ text: statement });
    return signer;
  }
}

export function signModernBadge(statement: string): cryptoFramework.Signature {
  const signer: TrustBadgeSigner = new TrustBadgeSigner();
  return signer.sign(statement);
}
```

### Rationale

trust 代码触发 @security/no-unsafe-ecdsa：ECDSA 签名/验签禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_ab28717fd41bd7c6`

ECDSA 签名/验签禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class InvoiceApprovalSigner {
  verify(invoice: string): cryptoFramework.Signature {
    const verifier = cryptoFramework.createVerify('ECC224|SHA1');
    verifier.init({ text: invoice });
    return verifier;
  }
}

export function legacyInvoiceVerify(invoice: string): cryptoFramework.Signature {
  const signer: InvoiceApprovalSigner = new InvoiceApprovalSigner();
  return signer.verify(invoice);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class InvoiceApprovalSigner {
  verify(invoice: string): cryptoFramework.Signature {
    const verifier = cryptoFramework.createVerify('ECC384|SHA256');
    verifier.init({ text: invoice });
    return verifier;
  }
}

export function secureInvoiceVerify(invoice: string): cryptoFramework.Signature {
  const signer: InvoiceApprovalSigner = new InvoiceApprovalSigner();
  return signer.verify(invoice);
}
```

### Rationale

invoice 代码触发 @security/no-unsafe-ecdsa：ECDSA 签名/验签禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_f653f74c909b48a6`

ECDSA 签名/验签禁止使用 SHA1。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordEcdsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC192|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyRecordEcdsa(payload: string): cryptoFramework.Signature {
  const signer: RecordEcdsaSigner = new RecordEcdsaSigner();
  return signer.execute(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordEcdsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC256|SHA384');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureRecordEcdsa(payload: string): cryptoFramework.Signature {
  const signer: RecordEcdsaSignerSecure = new RecordEcdsaSignerSecure();
  return signer.execute(payload);
}
```

### Rationale

record 代码触发 @security/no-unsafe-ecdsa：ECDSA 签名/验签禁止使用 SHA1。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
