import cryptoFramework from '@ohos.security.cryptoFramework';

class VehiclePairingKey {
  build(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC');
  }
}

export function legacyVehiclePairing(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: VehiclePairingKey = new VehiclePairingKey();
  return builder.build();
}
