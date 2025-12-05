import { huks } from '@kit.UniversalKeystoreKit';

class WeakHuksKey {
  async generate(alias: string): Promise<number> {
    const descriptor: string = 'ECB|SHA1|NONE';
    const options: huks.HuksOptions = this.buildOptions(descriptor);
    await huks.generateKeyItem(alias, options);
    return options.properties.length;
  }

  summarize(tags: Array<string>): string {
    return tags.join(':');
  }
  private buildOptions(descriptor: string): huks.HuksOptions {
    const lookup: Record<string, number> = {
      ECB: huks.HuksCipherMode.HUKS_MODE_ECB,
      SHA1: huks.HuksKeyDigest.HUKS_DIGEST_SHA1,
      NONE: huks.HuksKeyPadding.HUKS_PADDING_NONE
    };
    const [mode, digest, padding] = descriptor.split('|');
    const props: Array<huks.HuksParam> = [
        { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
        { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_128 },
        { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT },
        { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: lookup[mode] ?? huks.HuksCipherMode.HUKS_MODE_CBC },
        { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: lookup[digest] ?? huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
        { tag: huks.HuksTag.HUKS_TAG_PADDING, value: lookup[padding] ?? huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
    ];
    return { properties: props };
  }
}

export async function buildWeakHuksKey(alias: string): Promise<number> {
  const helper: WeakHuksKey = new WeakHuksKey();
  helper.summarize([alias, 'legacy']);
  return helper.generate(alias);
}
