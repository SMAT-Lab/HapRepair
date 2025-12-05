import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptRsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureReceipt(payload: string): cryptoFramework.Signature {
  const signer: ReceiptRsaSignerSecure = new ReceiptRsaSignerSecure();
  return signer.execute(payload);
}
