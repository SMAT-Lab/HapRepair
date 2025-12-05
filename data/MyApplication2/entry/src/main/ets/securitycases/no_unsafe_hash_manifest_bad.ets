import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeLegacyManifestHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('MD5');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
