import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionTicketAesChannelSecure {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|CBC|PKCS7');
    cipher.init({ key: 'session-ticket-secure' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapSecureSessionTicket(records: Array<string>): Uint8Array {
  const channel: SessionTicketAesChannelSecure = new SessionTicketAesChannelSecure();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
