import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureHasher {
  digest(data: string): string {
    const md = cryptoFramework.createMd('SHA256');
    md.init();
    md.update(data);
    return md.digest();
  }

  summarize(values: Array<number>): number {
    return values.reduce((acc, val) => acc + val, 0);
  }
}

export function hashSession(message: string): string {
  const hasher: SecureHasher = new SecureHasher();
  hasher.summarize([message.length, 10]);
  return hasher.digest(message);
}
