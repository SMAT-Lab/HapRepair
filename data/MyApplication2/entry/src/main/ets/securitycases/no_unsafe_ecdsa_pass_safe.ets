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
