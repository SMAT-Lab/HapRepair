import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionEnvelopeBuilderSecure {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA4096|PKCS1_OAEP|SHA256|MGF1_SHA256');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureSession(payload: string): cryptoFramework.Cipher {
  const builder: SessionEnvelopeBuilderSecure = new SessionEnvelopeBuilderSecure();
  return builder.seal(payload);
}
