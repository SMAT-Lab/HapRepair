import cryptoFramework from '@ohos.security.cryptoFramework';

export function legacyContractSign(payload: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA1024|PKCS1|SHA1');
  signer.init({ text: payload });
  return signer;
}
