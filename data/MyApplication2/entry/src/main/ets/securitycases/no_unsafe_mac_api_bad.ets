import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyApiMac(body: string): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA1');
  mac.init({ text: body });
  return mac;
}
