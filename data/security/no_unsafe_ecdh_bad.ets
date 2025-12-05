import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakEcdhFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('ECC');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-ecdh-weak-${pair?.publicKey ?? 'unknown'}`;
  }

  monitor(peers: Array<string>): string {
    return peers.join('|');
  }
}

export function issueWeakEcdh(alias: string): string {
  const factory: WeakEcdhFactory = new WeakEcdhFactory();
  factory.monitor([alias, 'legacy']);
  return factory.generate(alias);
}
