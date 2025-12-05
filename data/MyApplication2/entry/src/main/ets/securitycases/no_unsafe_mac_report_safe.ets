import cryptoFramework from '@ohos.security.cryptoFramework';

export function createSecureReportMac(body: Array<string>): cryptoFramework.Mac {
  const mac = cryptoFramework.createMac('SHA256');
  body.forEach((segment) => mac.update({ text: segment }));
  return mac;
}
