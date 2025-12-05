import cryptoFramework from '@ohos.security.cryptoFramework';

class ClusterDhExchangeSecure {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp4096');
    agreement.init({ peer: 'cluster-peer' });
    return this.annotate('secure', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateSecureCluster(): string {
  const exchange: ClusterDhExchangeSecure = new ClusterDhExchangeSecure();
  return exchange.start();
}
