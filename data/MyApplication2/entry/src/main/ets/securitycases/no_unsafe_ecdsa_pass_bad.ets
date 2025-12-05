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
