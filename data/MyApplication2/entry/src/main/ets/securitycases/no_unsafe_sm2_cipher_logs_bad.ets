import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsSm2Protector {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacyLogsSm2(payload: string): cryptoFramework.Cipher {
  const protector: LogsSm2Protector = new LogsSm2Protector();
  return protector.encrypt(payload);
}
