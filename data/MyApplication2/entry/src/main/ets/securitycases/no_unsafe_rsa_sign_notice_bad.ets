import cryptoFramework from '@ohos.security.cryptoFramework';

class NoticeRsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('RSA1536|PKCS1|MD5');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyNotice(payload: string): cryptoFramework.Signature {
  const signer: NoticeRsaSigner = new NoticeRsaSigner();
  return signer.execute(payload);
}
