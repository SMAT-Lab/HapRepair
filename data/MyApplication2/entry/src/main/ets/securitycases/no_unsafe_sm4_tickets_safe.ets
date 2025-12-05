import cryptoFramework from '@ohos.security.cryptoFramework';

class TicketsSm4ChannelSecure {
  seal(records: Array<string>): cryptoFramework.Cipher {
    const cipher = cryptoFramework.createCipher('SM4_128|CBC|PKCS7');
    cipher.init({ text: records.join('#') });
    return cipher;
  }
}

export function encryptSecureTicketsSm4(records: Array<string>): cryptoFramework.Cipher {
  const channel: TicketsSm4ChannelSecure = new TicketsSm4ChannelSecure();
  return channel.seal(records);
}
