import cryptoFramework from '@ohos.security.cryptoFramework';

class InvoiceApprovalSigner {
  verify(invoice: string): cryptoFramework.Signature {
    const verifier = cryptoFramework.createVerify('ECC384|SHA256');
    verifier.init({ text: invoice });
    return verifier;
  }
}

export function secureInvoiceVerify(invoice: string): cryptoFramework.Signature {
  const signer: InvoiceApprovalSigner = new InvoiceApprovalSigner();
  return signer.verify(invoice);
}
