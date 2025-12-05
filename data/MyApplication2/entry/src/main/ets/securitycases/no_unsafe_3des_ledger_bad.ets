import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyLedgerChannel {
  exportSnapshot(rows: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|PKCS5');
    cipher.init({ key: 'legacy-ledger' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(rows.join('#')));
  }
}

export function dumpLedger(rows: Array<string>): Uint8Array {
  const channel: LegacyLedgerChannel = new LegacyLedgerChannel();
  return channel.exportSnapshot(rows);
}
