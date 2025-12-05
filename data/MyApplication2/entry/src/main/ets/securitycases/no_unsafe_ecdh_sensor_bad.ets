import cryptoFramework from '@ohos.security.cryptoFramework';

class SensorEcdhKeyBuilder {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC192');
  }
}

export function buildLegacySensorEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: SensorEcdhKeyBuilder = new SensorEcdhKeyBuilder();
  return builder.generate();
}
