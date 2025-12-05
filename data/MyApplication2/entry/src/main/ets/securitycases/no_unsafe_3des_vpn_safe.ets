import cryptoFramework from '@ohos.security.cryptoFramework';

class RemoteTunnelBuilder {
  private peers: Array<string>;

  constructor(peers: Array<string>) {
    this.peers = peers;
  }

  establish(): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CFB|PKCS7');
    cipher.init({ key: 'remote-tunnel-2025' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(this.peers.join(',')));
  }
}

export function openModernVpn(peers: Array<string>): Uint8Array {
  const builder: RemoteTunnelBuilder = new RemoteTunnelBuilder(peers);
  return builder.establish();
}
