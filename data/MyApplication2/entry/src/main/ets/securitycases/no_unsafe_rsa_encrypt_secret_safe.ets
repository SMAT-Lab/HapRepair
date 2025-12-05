import cryptoFramework from '@ohos.security.cryptoFramework';

class SecretEnvelopeBuilderSecure {
  seal(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA3072|PKCS1_OAEP|SHA384|MGF1_SHA384');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function encryptSecureSecret(payload: string): cryptoFramework.Cipher {
  const builder: SecretEnvelopeBuilderSecure = new SecretEnvelopeBuilderSecure();
  return builder.seal(payload);
}
