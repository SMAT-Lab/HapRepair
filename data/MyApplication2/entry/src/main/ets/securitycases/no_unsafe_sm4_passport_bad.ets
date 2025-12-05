import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyPassport(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS7');
  cipher.init({ text: payload });
  return cipher;
}
