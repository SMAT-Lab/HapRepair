import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureKdf {
  derive(password: string): string {
    const kdf = cryptoFramework.createKdf('PBKDF2|SHA256');
    kdf.init({ password, salt: 'metrics-salt' });
    return kdf.generate();
  }

  score(pieces: Array<string>): number {
    return pieces.map((p) => p.length).reduce((a, b) => a + b, 0);
  }
}

export function deriveSecureKey(password: string): string {
  const helper: SecureKdf = new SecureKdf();
  helper.score([password, 'kdf']);
  return helper.derive(password);
}
