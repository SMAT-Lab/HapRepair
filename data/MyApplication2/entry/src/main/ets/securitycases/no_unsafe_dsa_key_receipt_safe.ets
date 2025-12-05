import cryptoFramework from '@ohos.security.cryptoFramework';

class ReceiptKeyPlantSecure {
  fabricate(): cryptoFramework.AsymmetricKeyGenerator {
    return cryptoFramework.createAsyKeyGenerator('DSA4096');
  }
}

export function issueSecureReceiptKey(): cryptoFramework.AsymmetricKeyGenerator {
  const factory: ReceiptKeyPlantSecure = new ReceiptKeyPlantSecure();
  return factory.fabricate();
}
