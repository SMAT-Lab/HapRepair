import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyTax(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA1');
  cipher.init({ text: payload });
  return cipher;
}
