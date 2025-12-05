import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureSm2KeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('SM2_256|SHA256');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-sm2-strong-${pair?.publicKey ?? 'unknown'}`;
  }

  record(alias: string): void {
    console.info(`sm2-strong-${alias}`);
  }
}

export function issueSafeSm2Key(alias: string): string {
  const factory: SecureSm2KeyFactory = new SecureSm2KeyFactory();
  factory.record(alias);
  return factory.generate(alias);
}
