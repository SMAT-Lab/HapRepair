import cryptoFramework from '@ohos.security.cryptoFramework';

export function deriveSecureArchiveKey(seed: string): cryptoFramework.Kdf {
  const kdf = cryptoFramework.createKdf('HKDF|SHA512');
  kdf.init({ text: seed });
  return kdf;
}
