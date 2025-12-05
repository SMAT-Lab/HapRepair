import cryptoFramework from '@ohos.security.cryptoFramework';

export function createLegacyLicenseMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA1');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
