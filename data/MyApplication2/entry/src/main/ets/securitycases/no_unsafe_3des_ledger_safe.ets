import cryptoFramework from '@ohos.security.cryptoFramework';

class ModernLedgerChannel {
  exportSnapshot(rows: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CBC|PKCS7');
    cipher.init({ key: 'modern-ledger' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(rows.join('#')));
  }
}

export function dumpLedger(rows: Array<string>): Uint8Array {
  const channel: ModernLedgerChannel = new ModernLedgerChannel();
  return channel.exportSnapshot(rows);
}
