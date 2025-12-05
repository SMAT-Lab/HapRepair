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
