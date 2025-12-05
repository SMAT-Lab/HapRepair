import cryptoFramework from '@ohos.security.cryptoFramework';

class DeviceOnboardingExchange {
  perform(deviceId: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1024');
    agreement.init({ peer: deviceId });
    return `${deviceId}:${agreement}`;
  }
}

export function onboardLegacyDevice(deviceId: string): string {
  const exchange: DeviceOnboardingExchange = new DeviceOnboardingExchange();
  return exchange.perform(deviceId);
}
