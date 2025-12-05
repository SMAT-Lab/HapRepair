import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeLegacyBackupHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA1');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
