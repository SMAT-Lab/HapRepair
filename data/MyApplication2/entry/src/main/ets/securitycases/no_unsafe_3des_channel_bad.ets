import cryptoFramework from '@ohos.security.cryptoFramework';

class ChannelTunnelBuilder {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|PKCS7');
    cipher.init({ key: 'channel-peers-legacy' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openLegacyChannelChannel(peers: Array<string>): Uint8Array {
  const builder: ChannelTunnelBuilder = new ChannelTunnelBuilder();
  builder.describe(peers);
  return builder.seal(peers);
}
