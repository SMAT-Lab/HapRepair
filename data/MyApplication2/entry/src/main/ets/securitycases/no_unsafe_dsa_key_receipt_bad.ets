import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptKeyPlant {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA1536');
  }
}

export function issueLegacyReceiptKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ReceiptKeyPlant = new ReceiptKeyPlant();
  return factory.fabricate();
}
