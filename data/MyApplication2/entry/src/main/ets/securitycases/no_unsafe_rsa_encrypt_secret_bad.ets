import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretEnvelopeBuilder {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA1536|PKCS1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptLegacySecret(payload: string): cryptoFramework.Cipher {
  const builder: SecretEnvelopeBuilder = new SecretEnvelopeBuilder();
  return builder.seal(payload);
}
