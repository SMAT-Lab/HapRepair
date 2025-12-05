import cryptoFramework from '@ohos.security.cryptoFramework';

class TrainingKeyFactory {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp1024');
  }
}

export function legacyTrainingKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: TrainingKeyFactory = new TrainingKeyFactory();
  return factory.generate();
}
