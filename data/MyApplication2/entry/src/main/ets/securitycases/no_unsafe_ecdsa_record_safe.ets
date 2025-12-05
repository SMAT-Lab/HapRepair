import cryptoFramework from '@ohos.security.cryptoFramework';

class RecordEcdsaSignerSecure {
  execute(payload: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC256|SHA384');
    signer.init({ text: payload });
    return signer;
  }
}

export function signSecureRecordEcdsa(payload: string): cryptoFramework.Signature {
  const signer: RecordEcdsaSignerSecure = new RecordEcdsaSignerSecure();
  return signer.execute(payload);
}
