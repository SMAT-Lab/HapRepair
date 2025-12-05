import { huks } from '@kit.UniversalKeystoreKit';

function createAccessCardKey(): void {
  let keyAlias: string = 'access-card';
  let properties: Array<huks.HuksParam> = [
    { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_RSA },
    { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_RSA_KEY_SIZE_1024 },
    { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_SIGN },
    { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
    { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
    { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
  ];
  let options: huks.HuksOptions = { properties: properties };
  huks.generateKeyItem(keyAlias, options);
}

createAccessCardKey();
