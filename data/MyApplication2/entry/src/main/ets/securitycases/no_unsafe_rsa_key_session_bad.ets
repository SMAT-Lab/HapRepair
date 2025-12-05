import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionRsaKeyPlant {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA1536|PRIMES_2');
  }
}

export function issueLegacySessionRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SessionRsaKeyPlant = new SessionRsaKeyPlant();
  return factory.produce();
}
