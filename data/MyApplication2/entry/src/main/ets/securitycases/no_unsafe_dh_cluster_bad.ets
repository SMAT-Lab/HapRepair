import cryptoFramework from '@ohos.security.cryptoFramework';

class ClusterDhExchange {
  start(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1024');
    agreement.init({ peer: 'cluster-peer' });
    return this.annotate('legacy', agreement);
  }

  annotate(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function negotiateLegacyCluster(): string {
  const exchange: ClusterDhExchange = new ClusterDhExchange();
  return exchange.start();
}
