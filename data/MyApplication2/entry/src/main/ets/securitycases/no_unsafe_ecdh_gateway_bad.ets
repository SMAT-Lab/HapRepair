import cryptoFramework from '@ohos.security.cryptoFramework';

class GatewayEcdhKeyBuilder {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC');
  }
}

export function buildLegacyGatewayEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: GatewayEcdhKeyBuilder = new GatewayEcdhKeyBuilder();
  return builder.generate();
}
