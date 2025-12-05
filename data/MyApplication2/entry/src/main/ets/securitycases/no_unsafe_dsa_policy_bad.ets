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
