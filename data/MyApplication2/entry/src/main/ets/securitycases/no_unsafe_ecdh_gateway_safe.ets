import cryptoFramework from '@ohos.security.cryptoFramework';

class GatewayEcdhKeyBuilderSecure {
  generate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('ECC256');
  }
}

export function buildSecureGatewayEcdh(): cryptoFramework.AsymmetricKeyGenerator {
  const builder: GatewayEcdhKeyBuilderSecure = new GatewayEcdhKeyBuilderSecure();
  return builder.generate();
}
