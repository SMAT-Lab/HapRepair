import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakKdf {
  derive(password: string): string {
    const kdf = cryptoFramework.createKdf('PBKDF2|SHA1');
    kdf.init({ password, salt: 'legacy-salt' });
    return kdf.generate();
  }

  score(pieces: Array<string>): number {
    return pieces.map((p) => p.length).reduce((a, b) => a + b, 0);
  }
}

export function deriveWeakKey(password: string): string {
  const helper: WeakKdf = new WeakKdf();
  helper.score([password]);
  return helper.derive(password);
}
