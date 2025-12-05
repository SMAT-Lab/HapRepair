import cryptoFramework from '@ohos.security.cryptoFramework';

class NoticeRsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA4096|PSS|SHA384|MGF1_SHA384');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureNotice(payload: string): cryptoFramework.Signature {
  const signer: NoticeRsaSignerSecure = new NoticeRsaSignerSecure();
  return signer.execute(payload);
}
