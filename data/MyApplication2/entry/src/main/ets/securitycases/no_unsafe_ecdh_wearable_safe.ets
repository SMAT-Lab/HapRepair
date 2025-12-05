import cryptoFramework from '@ohos.security.cryptoFramework';

class WearableKeyController {
  rotate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC384');
  }
}

export function secureWearableKey(): cryptoFramework.AsymmetricKeyGenerator {
  const controller: WearableKeyController = new WearableKeyController();
  return controller.rotate();
}
