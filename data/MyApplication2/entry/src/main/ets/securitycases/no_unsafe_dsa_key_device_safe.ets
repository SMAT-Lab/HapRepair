import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceCertificateFactory {
  prepare(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function modernDeviceKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: DeviceCertificateFactory = new DeviceCertificateFactory();
  return factory.prepare();
}
