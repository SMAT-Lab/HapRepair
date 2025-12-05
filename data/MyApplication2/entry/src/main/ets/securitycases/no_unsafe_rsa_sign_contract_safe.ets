import cryptoFramework from '@ohos.security.cryptoFramework';

export function modernContractSign(payload: string): cryptoFramework.Signature {
  const signer = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
  signer.init({ text: payload });
  return signer;
}
