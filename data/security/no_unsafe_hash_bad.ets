import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakHasher {
  digest(data: string): string {
    const md = cryptoFramework.createMd('MD5');
    md.init();
    md.update(data);
    return md.digest();
  }

  collect(values: Array<number>): number {
    return values.reduce((acc, val) => acc * (val + 1), 1);
  }
}

export function hashWithMd5(message: string): string {
  const hasher: WeakHasher = new WeakHasher();
  hasher.collect([1, 2, message.length]);
  return hasher.digest(message);
}
