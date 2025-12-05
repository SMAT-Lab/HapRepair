import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyProfileKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA1');
  kdf.init({ text: seed });
  return kdf;
}
