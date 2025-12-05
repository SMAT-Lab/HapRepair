import cryptoFramework from '@ohos.security.cryptoFramework';

class ProfileKeyFactory {
  make(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2');
  }
}

export function buildLegacyProfileKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ProfileKeyFactory = new ProfileKeyFactory();
  return factory.make();
}
