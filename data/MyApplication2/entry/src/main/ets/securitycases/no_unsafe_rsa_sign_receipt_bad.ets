import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptRsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA1024|PKCS1|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyReceipt(payload: string): cryptoFramework.Signature {
  const signer: ReceiptRsaSigner = new ReceiptRsaSigner();
  return signer.execute(payload);
}
