import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordsSm4Channel {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptLegacyRecordsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: RecordsSm4Channel = new RecordsSm4Channel();
  return channel.seal(records);
}
