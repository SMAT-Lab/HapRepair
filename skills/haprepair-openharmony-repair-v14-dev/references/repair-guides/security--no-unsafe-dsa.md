# @security/no-unsafe-dsa

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_02ed95e8428f9f50`

DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PolicyDsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('DSA1024|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyPolicyDsa(payload: string): cryptoFramework.Signature {
  const signer: PolicyDsaSigner = new PolicyDsaSigner();
  return signer.execute(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class PolicyDsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('DSA3072|SHA256');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecurePolicyDsa(payload: string): cryptoFramework.Signature {
  const signer: PolicyDsaSignerSecure = new PolicyDsaSignerSecure();
  return signer.execute(payload);
}
```

### Rationale

policy 代码触发 @security/no-unsafe-dsa：DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 2: `pair_05dbdcefb70beb58`

DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class AuditLogSigner {
  verify(payload: string): cryptoFramework.Signature {
    return cryptoFramework.createVerify('DSA2048|SHA1');
  }
}

export function legacyAuditVerify(payload: string): cryptoFramework.Signature {
  const verifier: AuditLogSigner = new AuditLogSigner();
  const verify = verifier.verify(payload);
  verify.init({ text: payload });
  return verify;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class AuditLogSigner {
  verify(payload: string): cryptoFramework.Signature {
    return cryptoFramework.createVerify('DSA3072|SHA256');
  }
}

export function secureAuditVerify(payload: string): cryptoFramework.Signature {
  const verifier: AuditLogSigner = new AuditLogSigner();
  const verify = verifier.verify(payload);
  verify.init({ text: payload });
  return verify;
}
```

### Rationale

log 代码触发 @security/no-unsafe-dsa：DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 3: `pair_92ce5b07bad45fb0`

DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LicenseSigner {
  sign(payload: string): cryptoFramework.Signature {
    return cryptoFramework.createSign('DSA1024|SHA1');
  }
}

export function legacyLicenseSignature(payload: string): cryptoFramework.Signature {
  const signer: LicenseSigner = new LicenseSigner();
  const signature = signer.sign(payload);
  signature.init({ text: payload });
  return signature;
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class LicenseSigner {
  sign(payload: string): cryptoFramework.Signature {
    return cryptoFramework.createSign('DSA3072|SHA256');
  }
}

export function modernLicenseSignature(payload: string): cryptoFramework.Signature {
  const signer: LicenseSigner = new LicenseSigner();
  const signature = signer.sign(payload);
  signature.init({ text: payload });
  return signature;
}
```

### Rationale

license 代码触发 @security/no-unsafe-dsa：DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 4: `pair_d16570223a409b42`

DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptDsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('DSA1536|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyReceiptDsa(payload: string): cryptoFramework.Signature {
  const signer: ReceiptDsaSigner = new ReceiptDsaSigner();
  return signer.execute(payload);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptDsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('DSA4096|SHA384');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureReceiptDsa(payload: string): cryptoFramework.Signature {
  const signer: ReceiptDsaSignerSecure = new ReceiptDsaSignerSecure();
  return signer.execute(payload);
}
```

### Rationale

receipt 代码触发 @security/no-unsafe-dsa：DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。

## Example 5: `pair_ee800b93ed469f08`

DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。

### Triggering pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakDsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('DSA1024|SHA1');
    signer.init({ key: 'legacy-dsa-key' });
    signer.update(data);
    return signer.sign();
  }

  detailLogs(entries: Array<string>): string {
    return entries.join(',');
  }
}

export function signWithWeakDsa(message: string): string {
  const signer: WeakDsaSigner = new WeakDsaSigner();
  signer.detailLogs(['audit', message]);
  return signer.sign(message);
}
```

### Repair pattern

```arkts
import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('DSA3072|SHA256');
    signer.init({ key: 'dsa-secure-key' });
    signer.update(data);
    return signer.sign();
  }

  summarizeData(changes: Array<string>): string {
    return changes.reverse().join('>');
  }
}

export function signInventory(message: string): string {
  const signer: SecureDsaSigner = new SecureDsaSigner();
  signer.summarizeData([message, message.toUpperCase()]);
  return signer.sign(message);
}
```

### Rationale

默认场景 代码触发 @security/no-unsafe-dsa：DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。
