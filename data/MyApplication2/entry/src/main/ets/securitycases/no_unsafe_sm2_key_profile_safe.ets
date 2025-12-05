import cryptoFramework from '@ohos.security.cryptoFramework';

class ProfileKeyFactory {
  make(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA384');
  }
}

export function buildSecureProfileKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ProfileKeyFactory = new ProfileKeyFactory();
  return factory.make();
}
