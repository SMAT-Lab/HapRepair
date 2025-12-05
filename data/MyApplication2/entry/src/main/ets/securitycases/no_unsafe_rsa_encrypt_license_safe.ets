import cryptoFramework from '@ohos.security.cryptoFramework';

class LicenseEnvelopeEncryptor {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA3072|PKCS1_OAEP|SHA256|MGF1_SHA256');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function buildModernLicenseEnvelope(payload: string): cryptoFramework.Cipher {
  const encryptor: LicenseEnvelopeEncryptor = new LicenseEnvelopeEncryptor();
  return encryptor.encrypt(payload);
}
