import cryptoFramework from '@ohos.security.cryptoFramework';

class TrainingKeyFactory {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp4096');
  }
}

export function secureTrainingKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: TrainingKeyFactory = new TrainingKeyFactory();
  return factory.generate();
}
