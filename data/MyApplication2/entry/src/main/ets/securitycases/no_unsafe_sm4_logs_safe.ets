import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureLogs(logs: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|CFB|PKCS7');
  cipher.init({ text: logs.join(';') });
  return cipher;
}
