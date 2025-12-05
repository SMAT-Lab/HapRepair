import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureEcdhFactory {
  generate(alias: string): string {
    const generator = cryptoFramework.createAsyKeyGenerator('ECC256');
    generator.init({ alias });
    const pair = generator.generateKeyPair();
    return `${alias}-ecdh-secure-${pair?.publicKey ?? 'unknown'}`;
  }

  monitor(peers: Array<string>): string {
    return peers.join('|');
  }
}

export function issueSafeEcdh(alias: string): string {
  const factory: SecureEcdhFactory = new SecureEcdhFactory();
  factory.monitor([alias]);
  return factory.generate(alias);
}
