import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakSm2Cipher {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|MD5');
    cipher.init({ key: 'sm2-enc-weak' });
    return cipher.update(payload);
  }

  decrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM2_256|MD5');
    cipher.init({ key: 'sm2-enc-weak' });
    return cipher.update(payload);
  }
}

export function runSm2WeakCipher(text: string): Uint8Array {
  const cipher: WeakSm2Cipher = new WeakSm2Cipher();
  const encoder: TextEncoder = new TextEncoder();
  const encrypted: Uint8Array = cipher.encrypt(encoder.encode(text));
  return cipher.decrypt(encrypted);
}
