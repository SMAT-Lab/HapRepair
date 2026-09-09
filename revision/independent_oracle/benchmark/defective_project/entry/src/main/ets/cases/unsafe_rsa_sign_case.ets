import cryptoFramework from '@ohos.security.cryptoFramework';

const exportSigner = cryptoFramework.createSign('RSA1024|PSS|SHA1|MGF1_SHA1');
