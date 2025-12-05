import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeSecureChecksum(parts: Array<string>): string {
  const digest = cryptoFramework.createMd('SHA512');
  parts.forEach((part) => digest.update({ text: part }));
  return `${digest}`;
}
