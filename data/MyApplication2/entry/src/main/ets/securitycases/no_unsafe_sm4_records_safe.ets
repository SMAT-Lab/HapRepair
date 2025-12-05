import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordsSm4ChannelSecure {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|CFB|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptSecureRecordsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: RecordsSm4ChannelSecure = new RecordsSm4ChannelSecure();
  return channel.seal(records);
}
