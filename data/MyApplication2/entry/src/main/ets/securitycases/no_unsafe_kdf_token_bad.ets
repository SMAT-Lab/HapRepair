import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveLegacyTokenKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('HKDF|SHA1');
  kdf.init({ text: seed });
  return kdf;
}
