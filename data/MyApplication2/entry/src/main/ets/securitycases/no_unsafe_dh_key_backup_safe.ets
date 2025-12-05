import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupDhKeyFactorySecure {
  create(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp8192');
  }
}

export function buildSecureBackupDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: BackupDhKeyFactorySecure = new BackupDhKeyFactorySecure();
  return factory.create();
}
