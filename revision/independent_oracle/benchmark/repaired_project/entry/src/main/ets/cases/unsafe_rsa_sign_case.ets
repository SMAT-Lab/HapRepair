import cryptoFramework from '@ohos.security.cryptoFramework';

const exportSigner = cryptoFramework.createSign('RSA3072|PSS|SHA256|MGF1_SHA256');
