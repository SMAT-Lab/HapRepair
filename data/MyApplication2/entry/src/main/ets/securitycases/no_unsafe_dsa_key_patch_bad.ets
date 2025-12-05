import cryptoFramework from '@ohos.security.cryptoFramework';

class PatchSigningKeyFactory {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA1024');
  }
}

export function legacyPatchKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PatchSigningKeyFactory = new PatchSigningKeyFactory();
  return factory.build();
}
