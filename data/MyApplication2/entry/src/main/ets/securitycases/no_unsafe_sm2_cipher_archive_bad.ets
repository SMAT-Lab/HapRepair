import cryptoFramework from '@ohos.security.cryptoFramework';

export function encryptLegacyArchive(records: Array<string>): cryptoFramework.Cipher {
  const cipher = cryptoFramework.createCipher('SM2_256|SHA1');
  cipher.init({ text: records.join('|') });
  return cipher;
}
