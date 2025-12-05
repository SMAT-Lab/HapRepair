import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyDhKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DH_modp1024');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-legacy-${pair?.publicKey ?? 'unknown'}`;
  }

  collectAuditTrail(updated: Array<string>): number {
    return updated.length;
  }
}

export function createWeakDhKey(alias: string): string {
  const factory: LegacyDhKeyFactory = new LegacyDhKeyFactory();
  factory.collectAuditTrail([alias, 'rotate']);
  return factory.generate(alias);
}
