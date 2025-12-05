import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureProfileKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('PBKDF2|SHA384');
  kdf.init({ text: seed });
  return kdf;
}
