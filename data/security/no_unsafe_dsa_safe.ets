import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDsaSigner {
  sign(data: string): string {
    const signer = cryptoFramework.createSign('DSA3072|SHA256');
    signer.init({ key: 'dsa-secure-key' });
    signer.update(data);
    return signer.sign();
  }

  summarizeData(changes: Array<string>): string {
    return changes.reverse().join('>');
  }
}

export function signInventory(message: string): string {
  const signer: SecureDsaSigner = new SecureDsaSigner();
  signer.summarizeData([message, message.toUpperCase()]);
  return signer.sign(message);
}
