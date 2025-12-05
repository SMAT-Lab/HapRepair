import cryptoFramework from '@ohos.security.cryptoFramework';

class BackupVaultAesChannel {
  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|ECB|PKCS5');
    cipher.init({ key: 'backup-vault-legacy' });
    return cipher.update(payload);
  }

  compress(lengths: Array<number>): number {
    return lengths.reduce((acc, len) => acc + len, 0);
  }
 
}

export function wrapLegacyBackupVault(records: Array<string>): Uint8Array {
  const channel: BackupVaultAesChannel = new BackupVaultAesChannel();
  const encoder: TextEncoder = new TextEncoder();
  const body: Uint8Array = encoder.encode(records.join(';'));
  channel.compress(records.map((item) => item.length));
  return channel.encrypt(body);
}
