import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecurePassport(payload: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|CBC|PKCS7');
  cipher.init({ text: payload });
  return cipher;
}
