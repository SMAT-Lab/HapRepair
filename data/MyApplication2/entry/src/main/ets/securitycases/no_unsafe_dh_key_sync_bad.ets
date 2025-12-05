import cryptoFramework from '@ohos.security.cryptoFramework';

class SyncKeyMaterial {
  refresh(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp1536');
  }
}

export function requestLegacyDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const material: SyncKeyMaterial = new SyncKeyMaterial();
  return material.refresh();
}
