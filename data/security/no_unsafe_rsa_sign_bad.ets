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
