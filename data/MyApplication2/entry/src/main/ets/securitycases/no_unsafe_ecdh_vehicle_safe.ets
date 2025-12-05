import cryptoFramework from '@ohos.security.cryptoFramework';

class VehiclePairingKey {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC256');
  }
}

export function secureVehiclePairing(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: VehiclePairingKey = new VehiclePairingKey();
  return builder.build();
}
