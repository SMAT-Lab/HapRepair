import cryptoFramework from '@ohos.security.cryptoFramework';

class ArchiveTunnelBuilder {
  seal(peers: Array<string>): Uint8Array {
    const cipher = cryptoFramework.createCipher('3DES|ECB|PKCS7');
    cipher.init({ key: 'archive-sync-legacy' });
    const encoder: TextEncoder = new TextEncoder();
    return cipher.update(encoder.encode(peers.join(',')));
  }

  describe(peers: Array<string>): number {
    return peers.join(',').length;
  }
}

export function openLegacyArchiveChannel(peers: Array<string>): Uint8Array {
  const builder: ArchiveTunnelBuilder = new ArchiveTunnelBuilder();
  builder.describe(peers);
  return builder.seal(peers);
}
