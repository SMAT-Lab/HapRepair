import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhExchangeSecure {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp6144');
    agreement.init({ peer: 'bridge-peer' });
    return this.annotate('secure', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateSecureBridge(): string {
  const exchange: BridgeDhExchangeSecure = new BridgeDhExchangeSecure();
  return exchange.start();
}
