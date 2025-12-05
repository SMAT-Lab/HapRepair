import cryptoFramework from '@ohos.security.cryptoFramework';

class LicenseEnvelopeEncryptor {
  encrypt(payload: string): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('RSA1024|PKCS1');
    cipher.init({ text: payload });
    return cipher;
  }
}

export function buildLegacyLicenseEnvelope(payload: string): cryptoFramework.Cipher {
  const encryptor: LicenseEnvelopeEncryptor = new LicenseEnvelopeEncryptor();
  return encryptor.encrypt(payload);
}
