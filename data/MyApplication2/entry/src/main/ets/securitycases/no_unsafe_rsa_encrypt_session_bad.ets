import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionEnvelopeBuilder {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA1024|PKCS1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacySession(payload: string): cryptoFramework.Cipher {
  const builder: SessionEnvelopeBuilder = new SessionEnvelopeBuilder();
  return builder.seal(payload);
}
