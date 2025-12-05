import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureBackupKey(secret: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA256');
  kdf.init({ text: secret });
  return kdf;
}
