import cryptoFramework from '@ohos.security.cryptoFramework';

export function modernLicenseKey(): cryptoFramework.AsymmetricKeyGenerator {
  return cryptoFramework.createAsyKeyGenerator('RSA4096|PRIMES_2');
}
