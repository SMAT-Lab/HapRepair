import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionRsaKeyPlantSecure {
  produce(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('RSA4096|PRIMES_2');
  }
}

export function issueSecureSessionRsaKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: SessionRsaKeyPlantSecure = new SessionRsaKeyPlantSecure();
  return factory.produce();
}
