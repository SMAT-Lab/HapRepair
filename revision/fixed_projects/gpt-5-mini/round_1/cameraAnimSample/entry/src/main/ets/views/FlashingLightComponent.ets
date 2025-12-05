/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { camera } from '@kit.CameraKit';
import Logger from '../common/utils/Logger';
import CameraService from '../mode/CameraService';

const TAG: string = 'FlashingLightComponent';

// 闪关灯页面
@Component
export struct FlashingLightComponent {
  // 闪光灯模式
  @Link @Watch('flashModeChange') flashMode: camera.FlashMode;
  @State flashIconResource: Resource = $r('app.media.ic_camera_public_flash_off');

  flashModeChange(): void {
    switch (this.flashMode) {
      case camera.FlashMode.FLASH_MODE_OPEN:
        this.flashIconResource = $r('app.media.ic_camera_public_flash_on');
        break;
      case camera.FlashMode.FLASH_MODE_AUTO:
        this.flashIconResource = $r('app.media.ic_camera_public_flash_auto');
        break;
      case camera.FlashMode.FLASH_MODE_ALWAYS_OPEN:
        this.flashIconResource = $r('app.media.flash_always_on');
        break;
      case camera.FlashMode.FLASH_MODE_CLOSE:
      default:
        this.flashIconResource = $r('app.media.ic_camera_public_flash_off');
    }
  }

  build() {
    Row() {
      Button() {
        Image(this.flashIconResource)
          .width($r('app.string.200px'))
          .height($r('app.string.200px'))
          .fillColor($r('app.color.white'))
      }
      .width($r('app.string.200px'))
      .height($r('app.string.200px'))
      .backgroundColor($r('app.color.flash_background_color'))
      .borderRadius($r('app.string.50px'))
      .onClick(() => {
        this.flashMode = (this.flashMode + 1) % 4;
        Logger.info(TAG, `flashMode: ${this.flashMode}`);
        CameraService.hasFlashFn(this.flashMode);
      })
    }
  }
}