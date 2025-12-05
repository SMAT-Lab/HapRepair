import cryptoFramework from '@ohos.security.cryptoFramework';

class ResultsKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA384');
  }
}

export function issueSecureResultsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ResultsKeyPlantSecure = new ResultsKeyPlantSecure();
  return factory.fabricate();
}
