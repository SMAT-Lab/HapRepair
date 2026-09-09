import cryptoFramework from '@ohos.security.cryptoFramework';

export function createLegacyReportCipher(): cryptoFramework.Cipher {
  return cryptoFramework.createCipher('3DES|ECB');
}
