import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureTax(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA256');
  cipher.init({ text: payload });
  return cipher;
}
