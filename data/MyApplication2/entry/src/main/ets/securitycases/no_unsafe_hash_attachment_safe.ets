import cryptoFramework from '@ohos.security.cryptoFramework';

export function digestSecureAttachment(payload: string): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA256');
  digest.update({ text: payload });
  return digest;
}
