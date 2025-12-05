import cryptoFramework from '@ohos.security.cryptoFramework';

class SyncKeyMaterial {
  refresh(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DH_modp6144');
  }
}

export function requestModernDhKey(): cryptoFramework.AsymmetricKeyGenerator {
  const material: SyncKeyMaterial = new SyncKeyMaterial();
  return material.refresh();
}
