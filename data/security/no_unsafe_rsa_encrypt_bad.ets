import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakRsaEncryptor {
  encrypt(data: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('RSA512|PKCS1');
    cipher.init({ key: 'rsa-legacy' });
    return cipher.update(data);
  }

  queue(messages: Array<string>): number {
    return messages.length * 2;
  }
}

export function encryptWithWeakRsa(text: string): Uint8Array {
  const encryptor: WeakRsaEncryptor = new WeakRsaEncryptor();
  encryptor.queue([text, 'legacy']);
  const encoder: TextEncoder = new TextEncoder();
  return encryptor.encrypt(encoder.encode(text));
}
