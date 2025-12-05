import cryptoFramework from '@ohos.security.cryptoFramework';

class SessionTicketAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|ECB|NoPadding');
    cipher.init({ key: 'session-ticket-legacy' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapLegacySessionTicket(records: Array<string>): Uint8Array {
  const channel: SessionTicketAesChannel = new SessionTicketAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
