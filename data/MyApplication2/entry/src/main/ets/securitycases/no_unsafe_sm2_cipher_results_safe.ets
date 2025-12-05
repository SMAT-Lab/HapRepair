import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsSm2ProtectorSecure {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM2_256|SHA512');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureResultsSm2(payload: string): cryptoFramework.Cipher {
  const protector: ResultsSm2ProtectorSecure = new ResultsSm2ProtectorSecure();
  return protector.encrypt(payload);
}
