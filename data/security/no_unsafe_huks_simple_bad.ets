import { huks } from '@kit.UniversalKeystoreKit';

const keyAlias: string = 'weakKeyAlias';
const properties: Array<huks.HuksParam> = [
  { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_ECC },
  { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_ECC_KEY_SIZE_256 },
  { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_SIGN |
      huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_VERIFY },
  { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
  { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
  { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
];

const options: huks.HuksOptions = {
  properties: properties
};

export async function createWeakHuksKey(): Promise<void> {
  await huks.generateKeyItem(keyAlias, options);
}
