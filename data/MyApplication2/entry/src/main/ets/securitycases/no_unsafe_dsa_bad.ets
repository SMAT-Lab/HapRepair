import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakDsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('DSA1024|SHA1');
    signer.init({ key: 'legacy-dsa-key' });
    signer.update(data);
    return signer.sign();
  }

  detailLogs(entries: Array<string>): string {
    return entries.join(',');
  }
}

export function signWithWeakDsa(message: string): string {
  const signer: WeakDsaSigner = new WeakDsaSigner();
  signer.detailLogs(['audit', message]);
  return signer.sign(message);
}
