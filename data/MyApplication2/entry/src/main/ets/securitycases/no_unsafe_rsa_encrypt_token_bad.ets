import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyToken(token: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('RSA512|PKCS1');
  cipher.init({ text: token });
  return cipher;
}
