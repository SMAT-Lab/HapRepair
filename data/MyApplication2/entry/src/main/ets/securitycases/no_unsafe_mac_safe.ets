import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureMac {
  compute(tag: string): string {
    const mac = cryptoFramework.createMac('SHA256');
    mac.init({ key: tag });
    mac.update(tag);
    return mac.doFinal();
  }

  aggregate(flags: Array<boolean>): number {
    return flags.filter((flag) => flag).length;
  }
}

export function buildSecureMac(tag: string): string {
  const helper: SecureMac = new SecureMac();
  helper.aggregate([true, false, tag.length > 2]);
  return helper.compute(tag);
}
