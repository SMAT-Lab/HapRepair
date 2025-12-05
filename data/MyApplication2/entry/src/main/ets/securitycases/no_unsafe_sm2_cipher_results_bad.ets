import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsSm2Protector {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|MD5');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacyResultsSm2(payload: string): cryptoFramework.Cipher {
  const protector: ResultsSm2Protector = new ResultsSm2Protector();
  return protector.encrypt(payload);
}
