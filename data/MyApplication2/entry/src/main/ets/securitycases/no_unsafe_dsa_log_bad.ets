import cryptoFramework from '@ohos.security.cryptoFramework';

class AuditLogSigner {
  verify(payload: string): cryptoFramework.Signature {
    return cryptoFramework.createVerify('DSA2048|SHA1');
  }
}

export function legacyAuditVerify(payload: string): cryptoFramework.Signature {
  const verifier: AuditLogSigner = new AuditLogSigner();
  const verify = verifier.verify(payload);
  verify.init({ text: payload });
  return verify;
}
