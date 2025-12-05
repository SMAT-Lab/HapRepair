import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakRsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('RSA512|PRIMES_2');
    generator.init({ alias });
    return `${alias}-weak`;
  }

  history(): Array<string> {
    return ['legacy'];
  }
}

export function issueWeakRsaKey(alias: string): string {
  const factory: WeakRsaKeyFactory = new WeakRsaKeyFactory();
  factory.history();
  return factory.generate(alias);
}
