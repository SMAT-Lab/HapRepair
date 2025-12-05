import cryptoFramework from '@ohos.security.cryptoFramework';

export function digestLegacyAttachment(payload: string): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('MD5');
  digest.update({ text: payload });
  return digest;
}
