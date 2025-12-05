import cryptoFramework from '@ohos.security.cryptoFramework';

class VideoStreamProtector {
  private frames: Array<Uint8Array> = [];

  constructor(streamId: string) {
    const encoder: TextEncoder = new TextEncoder();
    this.frames.push(encoder.encode(streamId));
  }

  cacheFrame(text: string): void {
    const encoder: TextEncoder = new TextEncoder();
    this.frames.push(encoder.encode(text));
  }

  sealChannel(): Uint8Array {
    const cipher = cryptoFramework.createCipher('AES256|ECB|PKCS5');
    cipher.init({ key: 'stream-legacy-key' });
    return cipher.update(this.combineFrames());
  }

  private combineFrames(): Uint8Array {
    const totalLength: number = this.frames.reduce((acc, frame) => acc + frame.length, 0);
    const merged: Uint8Array = new Uint8Array(totalLength);
    let offset: number = 0;
    this.frames.forEach((frame) => {
      merged.set(frame, offset);
      offset += frame.length;
    });
    return merged;
  }
}

export function buildLegacyVideoBundle(frames: Array<string>): Uint8Array {
  const protector: VideoStreamProtector = new VideoStreamProtector('lecture-channel');
  frames.forEach((frame) => protector.cacheFrame(frame));
  return protector.sealChannel();
}
