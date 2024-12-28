```arkts

/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
import display from '@ohos.display';
import window from '@ohos.window';
import { BusinessError } from '@ohos.base';
import {
  AvVideoPlayer,
  GlobalContext,
  IjkVideoPlayer,
  IVideoPlayer,
  LogUtils,
  PlayerType,
  PlayStatus,
  StandardGSYVideoModel
} from '@ohos/gsyvideoplayer';
import audio from '@ohos.multimedia.audio';
import emitter from '@ohos.events.emitter';
import settings from '@ohos.settings';
import {
  BaseDanmaku,
  BaseDanmakuParser,
  Callback,
  DANMAKU_STYLE_STROKEN,
  DanmakuContext,
  DanmakuTimer,
  DanmakuView,
  IDanmakus,
  IDanmakuView,
  JSONSource,
  OnDanmakuClickListener,
  Proxy,
  SpannedCacheStuffer,
  SystemClock
} from '@ohos/danmakuflamemaster';
import { BiliDanmukuParser } from './BiliDanmakuParser';
import { sourceData } from './DanmakuData';

let updateProgressTimer: number = 0;
let changeWidth: number = 0;
let changeHeight: number = 0;
let screenWidth: number = 0;
let screenHeight: number = 0;
let uiTime: number = 0;
let uiLockTime: number = 0;
let windowClass: window.Window | undefined = undefined;

let videoPlayEvent: emitter.InnerEvent = {
  eventId: 1
};
let videoInitEvent: emitter.InnerEvent = {
  eventId: 2
};
let videoPauseEvent: emitter.InnerEvent = {
  eventId: 3
};

let touchStartX: number = 0;
let touchStartY: number = 0;

@Component
export struct DanmakuVideoPlayer {
  mIVideoPlayer: IVideoPlayer | undefined = undefined;
  @Provide currentTime: string = "00:00";
  @Provide totalTime: string = "00:00";
  @State progressValue: number = 0;
  slideEnable: boolean = false;
  eventType: string = '';
  @State showBottomUi: boolean = false;
  @State fullShowTop: boolean = false;
  @State fullShowLock: boolean = false;
  @State lock: boolean = false;
  @State showPlay: boolean = true;
  @State showPause: boolean = false;
  mDestroyPage: boolean = false;
  @State loadingVisible: Visibility = Visibility.None;
  @State mDirection: number = 0;
  @State screenIsFull: boolean = false;
  @State danMuText: string = "弹幕关";
  // 是否显示音量数值的布局
  @State showVolumeUi: boolean = false;
  // 是否显示亮度数值的布局
  @State showBrightnessUi: boolean = false;
  // 是否显示横向滑动的进度布局
  @State showSeekProgressUi: boolean = false;
  // 当前音量 [0-20]
  @State mCurrentVolumeValue: number = 0;
  // 系统最大音量
  @State mMaxVolumeValue: number = 20;
  // 当前亮度 [0-255]
  @State mCurrentBrightness: number = 10;
  // 拖拽的时间
  @State mSeekCurrentTime: string = this.currentTime;
  // 当前进度
  @State mSeekTimePosition: number = 0;
  // 是否拖动进度条
  private isSeek: boolean = false;
  // 是否正在加载手势滑动禁止
  private isLoadNotSeek: boolean = false;
  // 当前拖拽进度的方向资源
  @State mSeekDirectionRes: Resource = $r('app.media.video_forward_icon');
  private mAudioManager: audio.AudioManager | null = null;
  private isVerticalDrag: boolean = false;
  private isHorizontalDrag: boolean = false;
  @State coverVisible: Visibility = Visibility.Visible;
  private panOptionBrightAndVolume: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.Vertical });
  private panOptionSeek: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.Horizontal });
  public title: string = "";
  private xComponentId: string = 'xid';
  @State ijkComponentVisible: Visibility = Visibility.Hidden;
  private danMuShow: boolean = true;
  sec: number = 3;
  PROGRESS_MAX_VALUE: number = 100;
  public videoModel: StandardGSYVideoModel = new StandardGSYVideoModel();
  @State
  model: DanmakuView.Model = new DanmakuView.Model()
  private mContext: DanmakuContext | undefined = undefined;
  private mParser: BaseDanmakuParser | undefined = undefined;
  private mCacheStufferAdapter: Proxy = new Pro()

  private createParser(): BaseDanmakuParser {
    let parser: BaseDanmakuParser = new BiliDanmukuParser();
    let jsonSource = new JSONSource(sourceData)
    parser.load(jsonSource)
    return parser;
  }

  videoInit: (iVideoPlayer: IVideoPlayer, xid: string) => void = (iVideoPlayer: IVideoPlayer, xid: string) => {
    LogUtils.getInstance().LOGI('VideoInit')
    this.mIVideoPlayer = iVideoPlayer
    this.xComponentId = xid;
    this.mIVideoPlayer.setUp(this.videoModel.getUrl(), this.videoModel.getCacheWithPlay())
  }

  private addDanmaku(isLive: Boolean) {
    if (this.mContext) {
      let danmaku: BaseDanmaku = this.mContext.mDanmakuFactory.createDanmaku(BaseDanmaku.TYPE_SCROLL_RL);
      danmaku.text = "这是一条弹幕" + SystemClock.uptimeMillis();
      danmaku.padding = 5;
      danmaku.priority = 0; // 可能会被各种过滤器过滤并隐藏显示
      danmaku.isLive = isLive.valueOf();
      danmaku.setTime(this.model.getCurrentTime() + 1200);
      if (this.mParser) {
        danmaku.textSize = 25 * (this.mParser.getDisplayer().getDensity() * 0.8);
      }
      danmaku.textColor = 0xffff0000;
      danmaku.textShadowColor = 0xffffffff;
      danmaku.borderColor = 0xff00ff00;
      this.model.addDanmaku(danmaku);
    }
  }

  @Builder
  topTitle() {
    Flex({ direction: FlexDirection.Row, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Start }) {
      Image($r('app.media.video_back'))
        .width(15)
        .height(15)
        .margin({ top: 10, left: 15 })
        .onClick(() => {
          this.videoModel.ExecuteBackClickListener();
        })
      Text(this.videoModel.getTitle())
        .fontSize('10vp')
        .margin({ top: 10, left: 5 })
        .fontColor($r('app.color.color_white'))
    }
    .align(Alignment.BottomEnd)
    .backgroundColor($r('app.color.bottom_controls_color'))
    .position({ x: 0, y: 0 })
    .padding(10)
  }

  @Builder
  middleControls() {
    Row() {
      Flex({ direction: FlexDirection.Row, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
      }
      .align(Alignment.BottomEnd)
      .width('33.3%')
      .height('100%')

      Flex({ direction: FlexDirection.Row, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
        if (this.showPlay) {
          Image($r('app.media.video_play_pressed'))
            .width(30)
            .height(30)
            .onClick(() => {
              this.videoToPlay();
            })
        }

        Image($r('app.media.icon_load'))
          .objectFit(ImageFit.Auto)
          .width(30)
          .height(30)
          .visibility(this.loadingVisible)
          .border({ width: 0 })
          .borderStyle(BorderStyle.Dashed)

        if (this.showPause) {
          Image($r('app.media.video_pause_normal'))
            .width(30)
            .height(30)
            .onClick(() => {
              LogUtils.getInstance().LOGI('click pause')
              this.videoToPause();
            })
        }
      }
      .align(Alignment.BottomEnd)
      .width('33.3%')
      .height('100%')

      Flex({ direction: FlexDirection.Row, alignItems: ItemAlign.Center, justifyContent: FlexAlign.End }) {
        if (this.mDirection == 1 && this.fullShowLock) {
          Image(this.lock ? $r('app.media.lock') : $r('app.media.unlock'))
            .width(20)
            .height(20)
            .objectFit(ImageFit.Contain)
            .margin({ right: 15 })
            .onClick(() => {
              this.lock = !this.lock;
              if (this.lock) {
                this.showBottomUi = false;
                this.fullShowTop = false;
                this.showPlay = false;
                this.showPause = false;
              }
            })
        }
      }
      .align(Alignment.BottomEnd)
      .width('33.3%')
      .height('100%')
    }.width("100%")
    .height('100%')
  }

  private videoToPause() {
    if (this.mIVideoPlayer) {
      LogUtils.getInstance().LOGI('standardGSYVideoPlayer onPause1 click')
      this.mIVideoPlayer.pause();
    }
  }

  private videoToPlay() {
    if (this.mIVideoPlayer) {
      LogUtils.getInstance().LOGI('standardGSYVideoPlayer onPlay click')
      this.timeCountdown();
      this.loadingVisible = Visibility.Visible;
      this.showPause = false;
      this.showPlay = false;
      this.videoJudgeToPlay();
    }
  }

  private videoJudgeToPlay() {
    LogUtils.getInstance().LOGI('videoJudgeToPlay');
    if (this.mIVideoPlayer) {
      if (GlobalContext.getContext().getObject('xid') !== this.mIVideoPlayer.xComponentId) {
        LogUtils.getInstance().LOGI('videoJudgeToPlay1');
        this.mIVideoPlayer.play();
        this.mIVideoPlayer.firstOrSeek = true;
      } else {
        LogUtils.getInstance().LOGI('videoJudgeToPlay2');
        if (this.mIVideoPlayer.playStatus == PlayStatus.PAUSE) {
          LogUtils.getInstance().LOGI('standardGSYVideoPlayer videoJudgeToPlay ResumePlay');
          this.mIVideoPlayer.resumePlay();
        } else {
          LogUtils.getInstance().LOGI('standardGSYVideoPlayer videoJudgeToPlay play: ' + this.mIVideoPlayer.videoUrl);
          this.mIVideoPlayer.play();
        }
      }
    }
  }

  @Builder
  bottomControls() {
    Column() {
      Flex({ direction: FlexDirection.Row, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Start }) {
        Text(this.currentTime)
          .width('80px')
          .fontSize('20px')
          .margin({ left: 20 })
          .fontColor($r('app.color.color_white'))

        Slider({
          value: this.progressValue,
          min: 0,
          max: this.PROGRESS_MAX_VALUE,
          step: 1,
          style: SliderStyle.OutSet
        })
          .blockColor(Color.Blue)
          .trackColor(Color.Gray)
          .selectedColor(Color.Blue)
          .showSteps(true)
          .showTips(true)
          .onChange((value: number, mode: SliderChangeMode) => {
            clearTimeout(uiTime);
            if (this.mIVideoPlayer) {
              switch (mode) {
                case SliderChangeMode.Begin:
                // 通知的回调会到的慢一点，导致下面设置的showPause = true无效
                // 使用isSeek参数去控制pause通知里面的状态操作
                  this.mIVideoPlayer.pause();
                  this.isSeek = true;
                  this.stopProgressTask();
                  this.showPlay = false;
                  this.showPause = true;
                  break;
                case SliderChangeMode.End:
                  this.showPause = false;
                  this.loadingVisible = Visibility.Visible;
                  let seekValue = value * (this.mIVideoPlayer.getDuration() / 100);
                  this.mIVideoPlayer.seekTo(seekValue);
                  this.model.seekTo(seekValue);
                  this.mIVideoPlayer.firstOrSeek = true;
                  LogUtils.getInstance().LOGI('slider-->seekValue end: ' + seekValue);
                  break;
              }
            }
          })
        Text(this.totalTime).width('80px').fontSize('20px').margin({ right: 5 }).fontColor($r('app.color.color_white'))

        Image(this.screenIsFull ? $r('app.media.video_shrink') : $r('app.media.video_enlarge'))
          .width(20)
          .height(18)
          .margin({ right: 15 })
          .onClick(() => {
            this.videoModel.ExecuteFullClickListener();
          })
      }.padding({ bottom: 10 })
      .align(Alignment.BottomEnd)
      .backgroundColor($r('app.color.bottom_controls_color'))

      Row() {
        Text("发送弹幕")
          .backgroundColor($r('app.color.color_add_danmu'))
          .width("90%")
          .textAlign(TextAlign.Center)
          .fontColor($r('app.color.color_add_text_danmu'))
          .onClick(() => {
            this.addDanmaku(false);
          })
          .layoutWeight(1)
        Text(this.danMuText)
          .textAlign(TextAlign.Center)
          .fontColor($r('app.color.color_show_danmu'))
          .onClick(() => {
            if (this.danMuShow) {
              this.model.hide();
              this.danMuText = '弹幕开';
            } else {
              this.model.setWidth(vp2px(changeWidth));
              this.model.setHeight(vp2px(changeHeight - 1));
              this.model.hide();
              this.model.show();
              this.danMuText = '弹幕关';
            }
            this.danMuShow = !this.danMuShow;
          })
      }.backgroundColor($r('app.color.bottom_controls_color')).margin({ bottom: (this.screenIsFull ? 10 : 0) })
    }
  }

  @Builder
  VolumeUi() {
    Column() {
      Progress({
        value: Number((this.mCurrentVolumeValue / this.mMaxVolumeValue * 100).toFixed(0)),
        total: 100,
        type: ProgressType.Linear
      })
        .style({ strokeWidth: 10, enableSmoothEffect: false })
        .rotate({ angle: -90 })
        .margin({top: 20, left: -20})
        .alignSelf(ItemAlign.Center)
        .backgroundColor(Color.White);

      Image($r('