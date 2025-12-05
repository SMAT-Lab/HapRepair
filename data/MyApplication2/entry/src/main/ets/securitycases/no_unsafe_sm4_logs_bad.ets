import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyLogs(logs: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS5');
  cipher.init({ text: logs.join(';') });
  return cipher;
}
