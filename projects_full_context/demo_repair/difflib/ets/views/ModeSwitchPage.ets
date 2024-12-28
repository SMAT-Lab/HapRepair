
/*
 * Copyright (c) 2023-2024 Huawei Device Co., Ltd.
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

// 反转摄像头_多机位_拍照_摄像

import { RecodeStopDialog } from '../Dialog/RecodeStopDialog';
import { MultiCameraDialog } from '../Dialog/MultiCameraDialog';
import DateTimeUtil from '../model/DateTimeUtil';
import Logger from '../model/Logger';
import CameraService from '../model/CameraService';
import image from '@ohos.multimedia.image';
import { BusinessError } from '@ohos.base';
import { Constants } from '../common/Constants';
import { GlobalContext } from '../common/GlobalContext';
import camera from '@ohos.multimedia.camera';
import { PromptAction } from '@ohos.arkui.UIContext';
import photoAccessHelper from '@ohos.file.photoAccessHelper';

const TAG: string = 'ModeSwitchPage';

@Component
export struct ModeSwitchPage {
  // 相机摄像头
  @Link cameraDeviceIndex: number;
  // surfaceID值
  @Prop @Watch('onSurfaceIdChange') surfaceId: string;
  @State localSurfaceId: string = '';

  onSurfaceIdChange(): void {
    this.localSurfaceId = this.surfaceId;
  }
  // 倒计时值
  @Link countdownNum: number;
...
              .backgroundColor(this.modelBagCol === 'macro' ? $r('app.color.theme_color') : '')
              .borderRadius(14)
              .onClick(async () => {
                this.modelBagCol = 'macro';
                AppStorage.setOrCreate('isHiddenSlide', false);
                CameraService.setCameraMode(8);
                await CameraService.initCamera(this.localSurfaceId, this.cameraDeviceIndex);
                let isSupported = CameraService.isMacroPhotoModeSupportedFn();
                if (!isSupported) {
                  (GlobalContext.get().getPromptAction() as PromptAction).showToast({
                    message: '当前不支持超级微距模式，请切换其他模式',
                    duration: 2000,
                    bottom: '50%'
                  })
                }
              })
              .visibility(this.deviceType === Constants.DEFAULT ? Visibility.Hidden : Visibility.Visible)

              // 拍照
              Column() {
                Text('拍照')
                  .fontSize(14)
                  .fontColor(Color.White)
              }
              .width(50)
              .backgroundColor(this.modelBagCol === 'photo' ? $r('app.color.theme_color') : '')
              .borderRadius(14)
              .onClick(async () => {
                this.modelBagCol = 'photo';
                AppStorage.setOrCreate('isHiddenSlide', false);
                CameraService.setCameraMode(0);
                await CameraService.initCamera(this.localSurfaceId, this.cameraDeviceIndex);
              })

              // 录像
              Column() {
                Text('录像')
                  .fontSize(14)
                  .fontColor(Color.White)
              }
              .width(50)
              .backgroundColor(this.modelBagCol === 'video' ? $r('app.color.theme_color') : '')
              .borderRadius(14)
              .onClick(async () => {
                this.modelBagCol = 'video';
                AppStorage.setOrCreate('isHiddenSlide', false);
                CameraService.setCameraMode(1);
                await CameraService.initCamera(this.localSurfaceId, this.cameraDeviceIndex);
              })

              // 多机位
              Column() {
                Text('多机位')
                  .fontSize(14)
                  .fontColor(Color.White)
              }
              .width(50)
              .backgroundColor(this.modelBagCol === 'multi' ? $r('app.color.theme_color') : '')
              .borderRadius(14)
              .onClick(async () => {
                this.modelBagCol = 'multi';
                CameraService.setCameraMode(0);
                await CameraService.initCamera(this.localSurfaceId, this.cameraDeviceIndex);
                this.MultiCameraDialogController.open();
              })
            }
            .height(28)
            .width('100%')
            .justifyContent(FlexAlign.SpaceBetween)
            .alignItems(VerticalAlign.Center)

            Row() {
              // 图库

              Row() {
                if (this.modelBagCol === 'video') {
                  Image(this.videoThumbnail || $r('app.media.pic_avatar_radio02'))
                    .aspectRatio(1)
                    .objectFit(ImageFit.Fill)
                    .border({ width: 2, color: 0xFFFFFF, radius: 40 })
                    .width('200px')
                    .height('200px')
                } else {
                  Image(this.imgThumbnail || $r('app.media.pic_avatar_radio02'))
                    .aspectRatio(1)
                    .objectFit(ImageFit.Fill)
                    .border({ width: 2, color: 0xFFFFFF, radius: 40 })
                    .width('200px')
                    .height('200px')
                }
              }.onClick(() => {
                let uri: string | PixelMap;
                if (this.modelBagCol === 'video') {
                  uri = this.videoUri;
                } else {
                  uri = this.imgThumbnail;
                }
                GlobalContext.get().getCameraSettingContext().startAbility({
                  parameters: { uri: uri },
                  action: 'ohos.want.action.viewData',
                  bundleName: 'com.huawei.hmos.photos',
                  abilityName: 'com.huawei.hmos.photos.MainAbility'
                });
              })
              

              // 拍照录像按键

              Row() {
                Button() {
                  Text()
                    .width('120px')
                    .height('120px')
                    .borderRadius('40px')
                    .backgroundColor(this.modelBagCol == 'video' || this.modelBagCol === 'superStab' ?
                    $r('app.color.theme_color') : Color.White)
                }
                .border({ width: 3, color: 0xFFFFFF, radius: 70 })
                .width('200px')
                .height('200px')
                .backgroundColor('rgba(255,255,255,0.20)')
                .onClick(() => {
                  // 倒计时拍照录像—默认拍照录像
                  this.countTakeVideoFn();
                })
              }
              

              // 前后置摄像头切换

              Row() {
                Button() {
                  Image($r('app.media.switch_camera'))
                    .width('120px').height('120px')
                }
                .width('200px')
                .height('200px')
                .backgroundColor('rgba(255,255,255,0.20)')
                .borderRadius('40px')
                .onClick(async () => {
                  // 切换摄像头
                  this.cameraDeviceIndex ? this.cameraDeviceIndex = 0 : this.cameraDeviceIndex = 1;
                  GlobalContext.get().setObject('cameraDeviceIndex', this.cameraDeviceIndex);
                  // 开始预览
                  await CameraService.initCamera(this.surfaceId, this.cameraDeviceIndex);
                })
              }
              
            }
            .width('100%')
            .justifyContent(FlexAlign.SpaceBetween)
            .alignItems(VerticalAlign.Center)
          }
          .justifyContent(FlexAlign.SpaceBetween)
          .height('20%')
          .padding({ left: 24, right: 24 })
          .margin({ bottom: 24 })
        }
      } else {
        Stack({ alignContent: Alignment.Bottom }) {
          Column({ space: 24 }) {
            Column() {
              Row() {
                Text().size({ width: 12, height: 12 }).backgroundColor($r('app.color.theme_color')).borderRadius(6)
                Text(this.dateTimeUtil.getVideoTime(this.videoRecodeTime))
                  .fontSize(30)
                  .fontColor(Color.White)
                  .margin({ left: 8 })
              }
            }
            .width('50%')

            Row() {
              Column() {
                // 录像抓拍键
                Button() {
                  Text().width('120px').height('120px').borderRadius('35px').backgroundColor(Color.White)
                }
                .border({ width: 2, color: 0xFFFFFF, radius: 45 })
                .width('200px')
                .height('200px')
                .backgroundColor('rgba(255,255,255,0.20)')
                .onClick(() => {
                  CameraService.takePicture();
                })
              }


              Row() {
                Column() {
                  // 录像停止键
                  Button() {
                    Image($r('app.media.ic_camera_video_close')).size({ width: 25, height: 25 })
                  }
                  .width('120px')
                  .height('120px')
                  .backgroundColor($r('app.color.theme_color'))
                  .onClick(() => {
                    if (this.timer !== undefined) {
                      clearInterval(this.timer);
                      Logger.debug(TAG, `onChangeRecord clear timer success`);
                    }
                    this.isRecording = !this.isRecording;
                    CameraService.stopVideo().then(async (fileAsset: photoAccessHelper.PhotoAsset): Promise<void> => {
                      try {
                        // 获取录制缩略图
                        this.videoThumbnail = await fileAsset.getThumbnail();
                        this.videoUri = fileAsset.uri;
                      } catch (error) {
                        let err = error as BusinessError;
                        Logger.error(TAG, `videoThumbnail err: ${JSON.stringify(err)}`);
                      }
                    })
                    // 打开停止弹框
                    this.recodeStopDialogController.open();
                  })
                }
                .width('180px')
                .height('180px')
                .borderRadius('60px')
                .backgroundColor($r('app.color.theme_color'))
                .justifyContent(FlexAlign.SpaceAround)
              }
              .justifyContent(FlexAlign.Center)
              .border({ width: 3, color: 0xFFFFFF, radius: 70 })
              .width('200px')
              .height('200px')
              .backgroundColor('rgba(255,255,255,0.20)')
              


              Row() {
                Column() {
                  // 录像暂停-播放键
                  Button() {
                    Image(!this.isRecording ? $r('app.media.ic_camera_video_on') : $r('app.media.ic_camera_video_off'))
                      .size({
                        width: 25,
                        height: 25
                      })
                  }
                  .width('120px')
                  .height('120px')
                  .backgroundColor($r('app.color.theme_color'))
                  .onClick(async () => {
                    await this.onChangeRecord();
                  })
                }
                .width('180px')
                .height('180px')
                .borderRadius('60px')
                .backgroundColor($r('app.color.theme_color'))
                .justifyContent(FlexAlign.SpaceAround)
              }
              .justifyContent(FlexAlign.Center)
              .border({ width: 3, color: 0xFFFFFF, radius: 70 })
              .width('200px')
              .height('200px')
              .backgroundColor('rgba(255,255,255,0.20)')
              
            }
            .width('100%')
            .justifyContent(FlexAlign.SpaceBetween)
            .alignItems(VerticalAlign.Center)
          }
          .justifyContent(FlexAlign.SpaceBetween)
          .height('20%')
          .padding({ left: 24, right: 24 })
          .margin({ bottom: 24 })
        }
      }
    }
    .position({ x: '0%', y: '75%' })
  }
}
