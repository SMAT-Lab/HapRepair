import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceOnboardingExchange {
  perform(deviceId: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp4096');
    agreement.init({ peer: deviceId });
    return `${deviceId}:${agreement}`;
  }
}

export function onboardSecureDevice(deviceId: string): string {
  const exchange: DeviceOnboardingExchange = new DeviceOnboardingExchange();
  return exchange.perform(deviceId);
}
