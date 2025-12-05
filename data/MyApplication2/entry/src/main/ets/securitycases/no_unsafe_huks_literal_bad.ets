import { huks } from '@kit.UniversalKeystoreKit';

class LiteralWeakHuksKey {
  async generate(alias: string): Promise<number> {
    const options: huks.HuksOptions = {
      properties: [
        { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
        { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_128 },
        { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT },
        { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_ECB },
        { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA1 },
        { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE }
      ]
    };
    await huks.generateKeyItem(alias, options);
    return options.properties.length;
  }

  describe(alias: string): string {
    return `${alias}-literal-weak`;
  }
}

export async function createLiteralWeakHuks(alias: string): Promise<number> {
  const factory: LiteralWeakHuksKey = new LiteralWeakHuksKey();
  factory.describe(alias);
  return factory.generate(alias);
}
