import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES128|GCM|PKCS5');
    cipher.init({ key: 'analytics-key-256' });
    return cipher.update(payload);
  }

  summarize(readings: Array<number>): number {
    return readings.reduce((acc, v) => acc + v, 0);
  }
}

export function protectTelemetry(message: string): Uint8Array {
  const channel: SecureAesChannel = new SecureAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(message);
  channel.summarize([1, 2, 3, message.length]);
  return channel.encrypt(body);
}
