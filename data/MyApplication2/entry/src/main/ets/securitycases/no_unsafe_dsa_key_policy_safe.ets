import cryptoFramework from '@ohos.security.cryptoFramework';

class PolicyKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function issueSecurePolicyKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PolicyKeyPlantSecure = new PolicyKeyPlantSecure();
  return factory.fabricate();
}
