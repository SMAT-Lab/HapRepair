import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakSm4Channel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM4|ECB|PKCS7');
    cipher.init({ key: 'sm4-legacy' });
    return cipher.update(payload);
  }

  digestFrames(frames: Array<number>): number {
    return frames.reduce((acc, val) => acc + val, 0);
  }
}

export function encryptWithWeakSm4(message: string): Uint8Array {
  const channel: WeakSm4Channel = new WeakSm4Channel();
  channel.digestFrames([message.length]);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(message));
}
