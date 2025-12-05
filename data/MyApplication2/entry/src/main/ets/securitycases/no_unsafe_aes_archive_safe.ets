import cryptoFramework from '@ohos.security.cryptoFramework';

class ResearchArchiveChannel {
  private departmentTag: string;

  constructor(tag: string) {
    this.departmentTag = tag;
  }

  encrypt(payload: Uint8Array): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES192|GCM|PKCS7');
    cipher.init({ key: `${this.departmentTag}-gcm-key` });
    return cipher.update(payload);
  }

  aggregateMarks(values: Array<number>): number {
    return values.reduce((acc, current) => acc + current, 0);
  }
}

export function archiveResearch(findings: Array<string>): Uint8Array {
  const encoder: TextEncoder = new TextEncoder();
  const channel: ResearchArchiveChannel = new ResearchArchiveChannel('bio-lab');
  const payload: Uint8Array = encoder.encode(findings.join('|'));
  channel.aggregateMarks([findings.length, payload.length]);
  return channel.encrypt(payload);
}
