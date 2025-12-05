import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretRsaKeyPlant {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA1024|PRIMES_2');
  }
}

export function issueLegacySecretRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SecretRsaKeyPlant = new SecretRsaKeyPlant();
  return factory.produce();
}
