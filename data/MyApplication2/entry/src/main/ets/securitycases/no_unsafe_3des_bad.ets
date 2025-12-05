import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyTripleDesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB');
    cipher.init({ key: '3des-legacy' });
    return cipher.update(payload);
  }

  audit(labels: Array<string>): string {
    return labels.join('-');
  }
}

export function encryptWithLegacy3Des(text: string): Uint8Array {
  const channel: LegacyTripleDesChannel = new LegacyTripleDesChannel();
  channel.audit(['legacy', text]);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(text));
}
