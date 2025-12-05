import cryptoFramework from '@ohos.security.cryptoFramework';

class WeakDhHandshake {
  negotiate(peerInfo: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1536');
    agreement.init({ peer: peerInfo });
    return `dh-weak-${peerInfo}`;
  }

  captureMetrics(events: Array<string>): number {
    return events.map((item) => item.length).reduce((a, b) => a + b, 0);
  }
}

export function startWeakDh(peer: string): string {
  const handshake: WeakDhHandshake = new WeakDhHandshake();
  handshake.captureMetrics([peer, 'handshake']);
  return handshake.negotiate(peer);
}
