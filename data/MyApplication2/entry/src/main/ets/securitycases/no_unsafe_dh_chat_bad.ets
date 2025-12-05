import cryptoFramework from '@ohos.security.cryptoFramework';

class LegacyChatKeyExchange {
  startHandshake(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp1536');
    agreement.init({ peer: 'support' });
    return this.makeTranscript('legacy', agreement);
  }

  private makeTranscript(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function legacyChatHandshake(): string {
  const exchange: LegacyChatKeyExchange = new LegacyChatKeyExchange();
  return exchange.startHandshake();
}
