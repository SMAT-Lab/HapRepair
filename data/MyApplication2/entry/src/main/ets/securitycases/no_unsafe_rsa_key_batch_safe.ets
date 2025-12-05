import cryptoFramework from '@ohos.security.cryptoFramework';

class BatchKeyGenerator {
  allocate(): Array<cryptoFramework.AsymmetricKeyGenerator> {
    return ['user-a', 'user-b'].map(() => cryptoFramework.createAsyKeyGenerator('RSA3072|PRIMES_2'));
  }
}

export function createModernBatch(): Array<cryptoFramework.AsymmetricKeyGenerator> {
  const generator: BatchKeyGenerator = new BatchKeyGenerator();
  return generator.allocate();
}
