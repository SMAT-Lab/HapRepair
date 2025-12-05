import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeSecureBackupHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA512');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
