import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureRsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('RSA3072|PRIMES_2');
    generator.init({ alias });
    return `${alias}-strong`;
  }

  history(): Array<string> {
    return ['init', 'rotate'];
  }
}

export function issueStrongRsaKey(alias: string): string {
  const factory: SecureRsaKeyFactory = new SecureRsaKeyFactory();
  factory.history();
  return factory.generate(alias);
}
