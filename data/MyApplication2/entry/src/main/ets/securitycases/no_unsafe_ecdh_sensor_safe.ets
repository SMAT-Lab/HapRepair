import cryptoFramework from '@ohos.security.cryptoFramework';

class SensorEcdhKeyBuilderSecure {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC384');
  }
}

export function buildSecureSensorEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: SensorEcdhKeyBuilderSecure = new SensorEcdhKeyBuilderSecure();
  return builder.generate();
}
