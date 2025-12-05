import cryptoFramework from '@ohos.security.cryptoFramework';

class TicketsSm4Channel {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|ECB|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptLegacyTicketsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: TicketsSm4Channel = new TicketsSm4Channel();
  return channel.seal(records);
}
