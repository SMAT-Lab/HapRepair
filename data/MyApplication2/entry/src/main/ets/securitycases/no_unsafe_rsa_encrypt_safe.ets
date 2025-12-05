import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureRsaEncryptor {
  encrypt(data: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('RSA3072|PKCS1_OAEP|SHA256|MGF1_SHA256');
    cipher.init({ key: 'rsa-strong-key' });
    return cipher.update(data);
  }

  queue(messages: Array<string>): number {
    return messages.length;
  }
}

export function encryptWithStrongRsa(text: string): Uint8Array {
  const encryptor: SecureRsaEncryptor = new SecureRsaEncryptor();
  encryptor.queue([text, text.slice(0, 2)]);
  const encoder: TextEncoder = new TextEncoder();
  return encryptor.encrypt(encoder.encode(text));
}
