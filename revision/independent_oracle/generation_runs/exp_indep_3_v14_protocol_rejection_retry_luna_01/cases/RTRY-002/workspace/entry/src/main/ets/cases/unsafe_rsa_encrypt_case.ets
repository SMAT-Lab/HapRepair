import cryptoFramework from '@ohos.security.cryptoFramework';

const exportCipher = cryptoFramework.createCipher('RSA2048|PKCS1_OAEP|SHA256|MGF1_SHA256');
