import cryptoFramework from '@ohos.security.cryptoFramework';

export function secureWebhookMac(events: Array<string>): string {
  const mac = cryptoFramework.createMac('SHA512');
  events.forEach((event) => mac.update({ text: event }));
  return `${mac}`;
}
