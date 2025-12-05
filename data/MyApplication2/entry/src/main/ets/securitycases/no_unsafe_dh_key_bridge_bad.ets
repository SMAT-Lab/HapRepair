import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhKeyFactory {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp1536');
  }
}

export function buildLegacyBridgeDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BridgeDhKeyFactory = new BridgeDhKeyFactory();
  return factory.create();
}
