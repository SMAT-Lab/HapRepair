import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakDsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DSA1024');
    generator.init({ alias });
    return `${alias}-weak`;
  }

  notify(alias: string): void {
    console.info(`issued-${alias}`);
  }
}

export function issueWeakDsaKey(alias: string): string {
  const factory: WeakDsaKeyFactory = new WeakDsaKeyFactory();
  factory.notify(alias);
  return factory.generate(alias);
}
