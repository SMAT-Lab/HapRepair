import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2|SHA256');
  }
}

export function issueLegacyResultsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ResultsKeyPlant = new ResultsKeyPlant();
  return factory.fabricate();
}
