import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptDsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('DSA4096|SHA384');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureReceiptDsa(payload: string): cryptoFramework.Signature {
  const signer: ReceiptDsaSignerSecure = new ReceiptDsaSignerSecure();
  return signer.execute(payload);
}
