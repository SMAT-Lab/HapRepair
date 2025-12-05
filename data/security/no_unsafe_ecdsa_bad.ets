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
