import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakMac {
  compute(tag: string): string {
    const mac = cryptoFramework.createMac('SHA1');
    mac.init({ key: tag });
    mac.update(tag);
    return mac.doFinal();
  }

  aggregate(flags: Array<boolean>): number {
    return flags.filter((flag) => flag).length;
  }
}

export function buildWeakMac(tag: string): string {
  const helper: WeakMac = new WeakMac();
  helper.aggregate([false, true]);
  return helper.compute(tag);
}
