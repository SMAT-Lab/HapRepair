import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptSecureArchive(records: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA512');
  cipher.init({ text: records.join('|') });
  return cipher;
}
