import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDsaKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DSA3072');
    generator.init({ alias });
    return `key-created-${alias}`;
  }

  record(alias: string): string {
    return `${alias}-recorded`;
  }
}

export function issueStrongDsaKey(alias: string): string {
  const factory: SecureDsaKeyFactory = new SecureDsaKeyFactory();
  factory.record(alias);
  return factory.generate(alias);
}
