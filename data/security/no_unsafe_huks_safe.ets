import huks from '@ohos.security.huks';

class SecureHuksKey {
  async generate(alias: string): Promise<number> {
    const properties: Array<huks.HuksParam> = [
      { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
      { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_256 },
      { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT | huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_DECRYPT },
      { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_CBC },
      { tag: huks.HuksTag.HUKS_TAG_DIGEST, value: huks.HuksKeyDigest.HUKS_DIGEST_SHA256 },
      { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_PKCS7 }
    ];
    const options: huks.HuksOptions = { properties };
    await huks.generateKeyItem(alias, options);
    return properties.length;
  }

  summarize(tags: Array<string>): string {
    return tags.join(':');
  }
}

export async function buildSecureHuksKey(alias: string): Promise<number> {
  const helper: SecureHuksKey = new SecureHuksKey();
  helper.summarize([alias, 'secure']);
  return helper.generate(alias);
}
