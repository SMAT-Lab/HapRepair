import cryptoFramework from '@ohos.security.cryptoFramework';

class PatchSigningKeyFactory {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function modernPatchKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: PatchSigningKeyFactory = new PatchSigningKeyFactory();
  return factory.build();
}
