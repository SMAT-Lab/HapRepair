import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyLicenseKey(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('RSA1024|PRIMES_2');
}
