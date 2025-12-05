import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2_256|SHA256');
  }
}

export function issueSecureLogsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: LogsKeyPlantSecure = new LogsKeyPlantSecure();
  return factory.fabricate();
}
