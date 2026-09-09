import cryptoFramework from '@ohos.security.cryptoFramework';

@Component
struct LegacyLocalStoreCipher {
  create(): void {
    cryptoFramework.createCipher('SM4_128|CBC|PKCS7');
  }

  build() {
    Text('legacy local store cipher')
  }
}
