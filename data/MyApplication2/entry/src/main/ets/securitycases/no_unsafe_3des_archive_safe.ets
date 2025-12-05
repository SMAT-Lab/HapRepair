import cryptoFramework from '@ohos.security.cryptoFramework';

class ArchiveTunnelBuilderSecure {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|CBC|PKCS7');
    cipher.init({ key: 'archive-sync-modern' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openSecureArchiveChannel(peers: Array<string>): Uint8Array {
  const builder: ArchiveTunnelBuilderSecure = new ArchiveTunnelBuilderSecure();
  builder.describe(peers);
  return builder.seal(peers);
}
