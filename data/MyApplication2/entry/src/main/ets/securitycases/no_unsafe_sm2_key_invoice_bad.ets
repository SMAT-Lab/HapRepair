import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacySm2Key(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('SM2|SHA256');
}
