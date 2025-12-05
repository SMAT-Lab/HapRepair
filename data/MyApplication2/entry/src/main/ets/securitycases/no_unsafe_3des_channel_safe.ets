import cryptoFramework from '@ohos.security.cryptoFramework';

class ChannelTunnelBuilderSecure {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CFB|PKCS7');
    cipher.init({ key: 'channel-peers-modern' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openSecureChannelChannel(peers: Array<string>): Uint8Array {
  const builder: ChannelTunnelBuilderSecure = new ChannelTunnelBuilderSecure();
  builder.describe(peers);
  return builder.seal(peers);
}
