import cryptoFramework from '@ohos.security.cryptoFramework';

export function modernUpdateSignature(changelog: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA4096|PSS|SHA384|MGF1_SHA384');
  signer.init({ text: changelog });
  return signer;
}
