import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsSm2ProtectorSecure {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureLogsSm2(payload: string): cryptoFramework.Cipher {
  const protector: LogsSm2ProtectorSecure = new LogsSm2ProtectorSecure();
  return protector.encrypt(payload);
}
