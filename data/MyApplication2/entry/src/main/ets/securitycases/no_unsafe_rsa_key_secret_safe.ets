import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretRsaKeyPlantSecure {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA3072|PRIMES_2');
  }
}

export function issueSecureSecretRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SecretRsaKeyPlantSecure = new SecretRsaKeyPlantSecure();
  return factory.produce();
}
