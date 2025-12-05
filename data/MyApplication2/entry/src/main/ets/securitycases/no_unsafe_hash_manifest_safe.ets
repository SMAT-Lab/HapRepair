import cryptoFramework from '@ohos.security.cryptoFramework';

export function computeSecureManifestHash(payload: Array<string>): cryptoFramework.Md {
  const digest = cryptoFramework.createMd('SHA256');
  payload.forEach((part) => digest.update({ text: part }));
  return digest;
}
