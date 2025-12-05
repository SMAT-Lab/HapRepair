import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptDsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('DSA1536|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyReceiptDsa(payload: string): cryptoFramework.Signature {
  const signer: ReceiptDsaSigner = new ReceiptDsaSigner();
  return signer.execute(payload);
}
