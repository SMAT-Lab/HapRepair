import cryptoFramework from '@ohos.security.cryptoFramework';

export function createSecureLicenseMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA512');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
