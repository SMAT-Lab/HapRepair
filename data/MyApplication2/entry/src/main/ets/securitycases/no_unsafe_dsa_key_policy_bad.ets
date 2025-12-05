import cryptoFramework from '@ohos.security.cryptoFramework';

class PolicyKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA2048');
  }
}

export function issueLegacyPolicyKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PolicyKeyPlant = new PolicyKeyPlant();
  return factory.fabricate();
}
