import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhKeyFactorySecure {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp6144');
  }
}

export function buildSecureBridgeDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BridgeDhKeyFactorySecure = new BridgeDhKeyFactorySecure();
  return factory.create();
}
