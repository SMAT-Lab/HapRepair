import cryptoFramework from '@ohos.security.cryptoFramework';

class TrustBadgeSigner {
  sign(statement: string): cryptoFramework.Signature {
    const signer = cryptoFramework.createSign('ECC224|SHA1');
    signer.init({ text: statement });
    return signer;
  }
}

export function signLegacyBadge(statement: string): cryptoFramework.Signature {
  const signer: TrustBadgeSigner = new TrustBadgeSigner();
  return signer.sign(statement);
}
