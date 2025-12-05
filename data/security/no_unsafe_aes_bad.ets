import cryptoFramework from '@ohos.security.cryptoFramework';

class InsecureAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES128|ECB|NoPadding');
    cipher.init({ key: 'legacy-key-128' });
    return cipher.update(payload);
  }

  checksum(values: Array<number>): number {
    return values.reduce((acc, val) => acc ^ val, 0);
  }
}

export function encryptWithLegacyMode(message: string): Uint8Array {
  const channel: InsecureAesChannel = new InsecureAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(message);
  channel.checksum([message.length, 42]);
  return channel.encrypt(body);
}
