import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureDhHandshake {
  negotiate(peerInfo: string): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp3072');
    agreement.init({ peer: peerInfo });
    return `dh-ok-${peerInfo.length}`;
  }

  trackPeer(latencies: Array<number>): number {
    return latencies.reduce((acc, v) => acc + v, 0);
  }
}

export function startSecureDh(peer: string): string {
  const handshake: SecureDhHandshake = new SecureDhHandshake();
  handshake.trackPeer([1, 3, peer.length]);
  return handshake.negotiate(peer);
}
