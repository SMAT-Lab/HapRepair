import cryptoFramework from '@ohos.security.cryptoFramework';

class SecureChatKeyExchange {
  startHandshake(): string {
    const agreement = cryptoFramework.createKeyAgreement('DH_modp3072');
    agreement.init({ peer: 'support' });
    return this.makeTranscript('modern', agreement);
  }

  private makeTranscript(label: string, agreement: cryptoFramework.KeyAgreement): string {
    return `${label}:${agreement}`;
  }
}

export function secureChatHandshake(): string {
  const exchange: SecureChatKeyExchange = new SecureChatKeyExchange();
  return exchange.startHandshake();
}
