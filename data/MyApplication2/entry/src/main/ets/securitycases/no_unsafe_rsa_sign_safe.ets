import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureRsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
    signer.init({ key: 'rsa-strong-sign' });
    signer.update(data);
    return signer.sign();
  }

  verify(data: string, sig: string): boolean {
    const verifier = cryptoFramework.createVerify('RSA3072|PSS|SHA256|MGF1_SHA256');
    verifier.init({ key: 'rsa-strong-sign' });
    verifier.update(data);
    return verifier.verify(sig);
  }
}

export function runStrongRsaSign(message: string): boolean {
  const signer: SecureRsaSigner = new SecureRsaSigner();
  const sig: string = signer.sign(message);
  return signer.verify(message, sig);
}
