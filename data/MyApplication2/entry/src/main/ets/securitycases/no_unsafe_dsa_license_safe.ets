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
