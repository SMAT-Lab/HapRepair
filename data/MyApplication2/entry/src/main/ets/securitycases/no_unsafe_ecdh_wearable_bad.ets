import cryptoFramework from '@ohos.security.cryptoFramework';

class WearableKeyController {
  rotate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC192');
  }
}

export function legacyWearableKey(): cryptoFramework.AsymmetricKeyGenerator {
  const controller: WearableKeyController = new WearableKeyController();
  return controller.rotate();
}
