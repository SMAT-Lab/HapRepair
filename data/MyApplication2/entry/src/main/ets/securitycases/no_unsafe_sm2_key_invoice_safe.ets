import cryptoFramework from '@ohos.security.cryptoFramework';

export function secureSm2Key(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA256');
}
