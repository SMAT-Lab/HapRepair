import cryptoFramework from '@ohos.security.cryptoFramework';

class LogsKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('SM2');
  }
}

export function issueLegacyLogsKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: LogsKeyPlant = new LogsKeyPlant();
  return factory.fabricate();
}
