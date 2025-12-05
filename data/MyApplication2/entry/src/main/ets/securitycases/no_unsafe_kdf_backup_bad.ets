import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyBackupKey(secret: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA1');
  kdf.init({ text: secret });
  return kdf;
}
