import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureEcdsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('ECC256|SHA256');
    signer.init({ key: 'ecc-safe-key' });
    signer.update(data);
    return signer.sign();
  }

  verifySignature(data: string, sig: string): boolean {
    const verifier = cryptoFramework.createVerify('ECC256|SHA256');
    verifier.init({ key: 'ecc-safe-key' });
    verifier.update(data);
    return verifier.verify(sig);
  }
}

export function runSecureEcdsa(message: string): boolean {
  const handler: SecureEcdsaSigner = new SecureEcdsaSigner();
  const sig: string = handler.sign(message);
  return handler.verifySignature(message, sig);
}
