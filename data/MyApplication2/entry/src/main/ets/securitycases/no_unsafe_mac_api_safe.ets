import cryptoFramework from '@ohos.security.cryptoFramework';

export function secureApiMac(body: string): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA256');
  mac.init({ text: body });
  return mac;
}
