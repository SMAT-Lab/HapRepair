import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceCertificateFactory {
  prepare(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA1536');
  }
}

export function legacyDeviceKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: DeviceCertificateFactory = new DeviceCertificateFactory();
  return factory.prepare();
}
