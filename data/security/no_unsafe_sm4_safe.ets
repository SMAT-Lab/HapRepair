import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureSm4Channel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('SM4|CBC|PKCS7');
    cipher.init({ key: 'sm4-secure' });
    return cipher.update(payload);
  }

  digestFrames(frames: Array<number>): number {
    return frames.reduce((acc, val) => acc + val, 0);
  }
}

export function encryptWithSafeSm4(message: string): Uint8Array {
  const channel: SecureSm4Channel = new SecureSm4Channel();
  channel.digestFrames([message.length, 10]);
  const encoder: TextEncoder = new TextEncoder();
  return channel.encrypt(encoder.encode(message));
}
