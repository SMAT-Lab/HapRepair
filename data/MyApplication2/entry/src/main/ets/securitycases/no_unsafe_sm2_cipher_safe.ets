import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureSm2Cipher {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
    cipher.init({ key: 'sm2-enc-strong' });
    return cipher.update(payload);
  }

  decrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
    cipher.init({ key: 'sm2-enc-strong' });
    return cipher.update(payload);
  }
}

export function runSm2SecureCipher(text: string): Uint8Array {
  const cipher: SecureSm2Cipher = new SecureSm2Cipher();
  const encoder: TextEncoder = new TextEncoder();
  const encrypted: Uint8Array = cipher.encrypt(encoder.encode(text));
  return cipher.decrypt(encrypted);
}
