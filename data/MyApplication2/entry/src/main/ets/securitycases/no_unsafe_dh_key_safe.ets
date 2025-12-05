import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDhKeyFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('DH_modp3072');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-secure-${pair?.publicKey ?? 'unknown'}`;
  }

  rotateHistory(oldKeys: Array<string>): string {
    return oldKeys.join('|');
  }
}

export function createStrongDhKey(alias: string): string {
  const factory: SecureDhKeyFactory = new SecureDhKeyFactory();
  factory.rotateHistory([alias, 'previous']);
  return factory.generate(alias);
}
