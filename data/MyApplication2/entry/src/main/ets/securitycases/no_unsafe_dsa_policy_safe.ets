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
