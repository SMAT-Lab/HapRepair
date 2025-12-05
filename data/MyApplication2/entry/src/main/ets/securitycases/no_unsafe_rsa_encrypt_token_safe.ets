import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureToken(token: string): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('RSA4096|PKCS1_OAEP|SHA256|MGF1_SHA256');
  cipher.init({ text: token });
  return cipher;
}
