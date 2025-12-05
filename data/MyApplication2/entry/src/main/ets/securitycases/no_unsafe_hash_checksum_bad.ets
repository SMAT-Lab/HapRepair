import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeLegacyChecksum(parts: Array<string>): string {
  const digest = cryptoFramework.createMd('SHA1');
  parts.forEach((part) => digest.update({ text: part }));
  return `${digest}`;
}
