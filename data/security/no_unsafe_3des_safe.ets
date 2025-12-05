import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureTripleDesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CBC|PKCS7');
    cipher.init({ key: '3des-secure' });
    return cipher.update(payload);
  }

  audit(labels: Array<string>): string {
    return labels.sort().join('-');
  }
}

export function encryptWithSafe3Des(text: string): Uint8Array {
  const channel: SecureTripleDesChannel = new SecureTripleDesChannel();
  channel.audit(['alpha', 'beta']);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(text));
}
