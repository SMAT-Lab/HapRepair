import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyWebhookMac(events: Array<string>): string {
  const mac = cryptoFramework.createMac('SHA1');
  events.forEach((event) => mac.update({ text: event }));
  return `${mac}`;
}
