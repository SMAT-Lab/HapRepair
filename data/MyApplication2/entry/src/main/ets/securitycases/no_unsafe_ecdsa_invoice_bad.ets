import cryptoFramework from '@ohos.security.cryptoFramework';

class InvoiceApprovalSigner {
  verify(invoice: string): cryptoFramework.Signature {
    const verifier = cryptoFramework.createVerify('ECC224|SHA1');
    verifier.init({ text: invoice });
    return verifier;
  }
}

export function legacyInvoiceVerify(invoice: string): cryptoFramework.Signature {
  const signer: InvoiceApprovalSigner = new InvoiceApprovalSigner();
  return signer.verify(invoice);
}
