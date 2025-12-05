import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupDhKeyFactory {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp2048');
  }
}

export function buildLegacyBackupDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BackupDhKeyFactory = new BackupDhKeyFactory();
  return factory.create();
}
