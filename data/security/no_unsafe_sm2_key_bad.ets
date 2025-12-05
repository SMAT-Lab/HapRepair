import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakSm2KeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('SM2|MD5');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-sm2-weak-${pair?.publicKey ?? 'unknown'}`;
  }

  record(alias: string): void {
    console.info(`sm2-weak-${alias}`);
  }
}

export function issueWeakSm2Key(alias: string): string {
  const factory: WeakSm2KeyFactory = new WeakSm2KeyFactory();
  factory.record(alias);
  return factory.generate(alias);
}
