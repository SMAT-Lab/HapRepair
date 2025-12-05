import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordEcdsaSigner {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC192|SHA1');
    signer.init({ text: payload });
    return signer;
  }
}

export function signLegacyRecordEcdsa(payload: string): cryptoFramework.Signature {
  const signer: RecordEcdsaSigner = new RecordEcdsaSigner();
  return signer.execute(payload);
}
