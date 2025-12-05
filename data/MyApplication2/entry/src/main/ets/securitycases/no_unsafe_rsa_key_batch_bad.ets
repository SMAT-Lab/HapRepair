import cryptoFramework from '@ohos.security.cryptoFramework';

class BatchKeyGenerator {
  allocate(): Array<cryptoFramework.AsymmetricKeyGenerator> {
    return ['user-a', 'user-b'].map(() => cryptoFramework.createAsyKeyGenerator('RSA1536|PRIMES_2'));
  }
}

export function createLegacyBatch(): Array<cryptoFramework.AsymmetricKeyGenerator> {
  const generator: BatchKeyGenerator = new BatchKeyGenerator();
  return generator.allocate();
}
