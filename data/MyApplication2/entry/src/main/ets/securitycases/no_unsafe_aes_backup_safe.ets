import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupVaultAesChannelSecure {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|GCM|PKCS7');
    cipher.init({ key: 'backup-vault-secure' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapSecureBackupVault(records: Array<string>): Uint8Array {
  const channel: BackupVaultAesChannelSecure = new BackupVaultAesChannelSecure();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
