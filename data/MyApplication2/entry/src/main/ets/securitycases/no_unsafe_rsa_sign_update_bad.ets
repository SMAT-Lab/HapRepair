import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyUpdateSignature(changelog: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA1536|PKCS1|MD5');
  signer.init({ text: changelog });
  return signer;
}
