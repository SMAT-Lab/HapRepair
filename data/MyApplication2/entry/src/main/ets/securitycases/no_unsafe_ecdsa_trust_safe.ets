import cryptoFramework from '@ohos.security.cryptoFramework';

class TrustBadgeSigner {
  sign(statement: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC256|SHA256');
    signer.init({ text: statement });
    return signer;
  }
}

export function signModernBadge(statement: string): cryptoFramework.Signature {
  const signer: TrustBadgeSigner = new TrustBadgeSigner();
  return signer.sign(statement);
}
