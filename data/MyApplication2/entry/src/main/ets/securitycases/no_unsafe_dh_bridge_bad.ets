import cryptoFramework from '@ohos.security.cryptoFramework';

class BridgeDhExchange {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1536');
    agreement.init({ peer: 'bridge-peer' });
    return this.annotate('legacy', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateLegacyBridge(): string {
  const exchange: BridgeDhExchange = new BridgeDhExchange();
  return exchange.start();
}
