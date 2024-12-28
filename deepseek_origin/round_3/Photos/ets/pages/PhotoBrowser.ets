```arkts

/*
 * Copyright (c) 2023 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import router, { ParamsInterface } from '@system.router';
import Want from '@ohos.application.Want';
import { Log } from '../utils/Log';
import { Action } from '../models/Action';
import { MenuOperation } from '../menus/MenuOperation';
import { MenuContext } from '../menus/MenuContext';
import {
  GotoPhotosMenuOperation
} from '../menus/GotoPhotosMenuOperation';
import { BatchDeleteMenuOperation } from '../menus/BatchDeleteMenuOperation';
import { BatchRecoverMenuOperation } from '../menus/BatchRecoverMenuOperation';
import { ClearRecycleMenuOperation } from '../menus/ClearRecycleMenuOperation';
import { MediaConstants } from '../constants/MediaConstants';
import { MoveMenuOperation } from '../menus/MoveMenuOperation';
import { DateUtil } from '../utils/DateUtil';
import { CustomDialogView } from '../components/CustomDialogView';
import { Constants } from '../constants/Constants';
import { Broadcast } from '../common/Broadcast';
import { BroadcastConstants } from '../constants/BroadcastConstants';
import { broadcastManager } from '../common/BroadcastManager';
import { Constants as PhotoConstants } from '../constants/BrowserConstants';
import { PhotoBrowserBg } from '../components/PhotoBrowserBg';
import { PhotoBrowserActionBar } from '../components/PhotoBrowserActionBar';
import { ToolBar } from '../components/ToolBar';
import { PhotoSwiper } from '../components/PhotoSwiper';
import { screenManager } from '../common/ScreenManager';
import { JumpSourceToMain } from '../models/JumpSourceToMain';
import { RenameMenuOperation } from '../menus/RenameMenuOperation';
import { MediaOperationType } from '../models/MediaOperationType';
import { showToast } from '../utils/UiUtil';
import { getResourceString } from '../utils/ResourceUtils';
import { CommonObserverCallback } from '../common/CommonObserverCallback';
import { userFileObserver } from '../base/UserFileObserver';
import { mMultimodalInputManager } from '../common/MultimodalInputManager';
import { GroupItemDataSource } from '../common/GroupItemDataSource';
import { UserFileDataItem } from '../base/UserFileDataItem';
import { InnerUserFileDataItem } from '../base/InnerUserFileDataItem';
import { SimpleAlbumDataItem } from '../common/SimpleAlbumDataItem';
import { getAlbumDisplayName } from '../base/UserFileDataHelper';
import { startAbility, terminateSelf } from '../utils/AbilityUtils';
import { userFileDataItemCache } from '../base/UserFileDataItemCache';
import { GlobalContext } from '../common/GlobalContext';
// page of large photo

const TAG = 'PhotoBrowser';

@Entry
@Component
struct PhotoBrowser {
  mOpacity: number = 1;
  @Provide browserBackgroundColor: Resource = $r('app.color.default_background_color');
  @Provide('dateTitle') photoDate: string = '';
  @Provide('timeLocationTitle') timeAndLocation: string = '';
  @Provide menuList: Action[] = [];
  @State toolMenuList: Action[] = [];
  @Provide topMenuList: Action[] = [];
  @Provide moreMenuList: Action[] = [];
  @Provide broadCast: Broadcast = new Broadcast();
  @Provide isShowBar: boolean = true;
  @Provide isPullingDown: boolean = false;
  @Provide pageFrom: number = Constants.ENTRY_FROM.NORMAL;
  @Provide canSwipe: boolean = false;
  @State @Watch('currentShowChange') currentShow: boolean = true;
  @StorageLink('isHorizontal') isHorizontal: boolean = screenManager.isHorizontal();
  @StorageLink('TimelinePageIndex') TimelinePageIndex: number = Constants.INVALID;
  @StorageLink('PhotoGridPageIndex') PhotoGridPageIndex: number = Constants.INVALID;
  @StorageLink('isSplitMode') isSplitMode: boolean = screenManager.isSplitMode();
  @StorageLink('leftBlank') leftBlank: number[] = [0, 0, 0, 0];
  @StorageLink('entryFromHap') entryFromHap: number = Constants.ENTRY_FROM_NONE;
  photoBrowserTransition: string = '';
  @State isDataChanged:boolean = true;
  private isFromCard = false;

  // swiper currentIndex, there may not be onChanged callback during data refresh, so mediaItem cannot be saved
  @Provide('transitionIndex') currentIndex: number = 0;
  controller: SwiperController = new SwiperController();
  private dataObserver: CommonObserverCallback = new CommonObserverCallback(this);

  // The global Broadcast of the application process. Event registration and destruction should be paired
  private appBroadcast: Broadcast = broadcastManager.getBroadcast();
  private isFromCamera = false;
  private isFromViewData = false;

  // the source of jump to the index page
  private jumpSourceToMain: number = JumpSourceToMain.None;

  // time to view the current picture
  private checkedTransition: string = '';

  // When clicking quickly, only run aboutToAppear for the first time
  private hasAppeared: boolean = false;
  private albumInfo: SimpleAlbumDataItem = new SimpleAlbumDataItem('','','','','',0,0);
  private deviceName = '';
  private backFromCopy = false;
  private pullDownFlag: boolean = false;
  private browserDataSource: GroupItemDataSource = new GroupItemDataSource()
  private isToEdit = false;
  private globalThis = GlobalContext.getContext();

  onMenuClicked(action: Action): void {
    this.onMenuClickedBindImpl(action)
  }

  private onMenuClickedBindImpl(action: Action): void {
    Log.info(TAG, 'onMenuClicked, action: ' + action.actionID + ', isPullingDown: ' + this.isPullingDown);
    if (this.isPullingDown) {
      return;
    }
    let menuOperation: MenuOperation;
    let menuContext: MenuContext = new MenuContext();
    let currentPhoto = this.getCurrentPhoto();
    if (action === Action.BACK) {
      this.onBackPress();
      return;
    } else if (action === Action.INFO) {
      if (currentPhoto == undefined) {
        return;
      }
      this.broadCast.emit(BroadcastConstants.SHOW_DETAIL_DIALOG, [currentPhoto]);
      return;
    } else if (action === Action.NOT_FAVORITE || action === Action.FAVORITE) {
      if (currentPhoto == undefined) {
        return;
      }
      currentPhoto.setFavor().then<void, void>((isSuccess: boolean): void => { return this.setFavourite(isSuccess) })
      return;
    } else if (action === Action.DELETE) {
      if (currentPhoto == undefined) {
        return;
      }
      menuContext.withItems([currentPhoto])
        .withDeletePageFromType(BroadcastConstants.DELETE_FROM_BROWSER)
        .withAlbumId( Boolean<SimpleAlbumDataItem>(this.albumInfo) ? ( Boolean<string>(this.albumInfo.id) ? this.albumInfo.id : this.albumInfo.toString() ) : '')
        .withBroadCast(this.broadCast)
        .withOperationEndCallback((): void => this.onDeleteEndBindImpl())
      menuOperation = new BatchDeleteMenuOperation(menuContext);
    } else if (action === Action.CLEAR_RECYCLE) {
      if (currentPhoto == undefined) {
        return;
      }
      menuContext.withItems([currentPhoto])
        .withBroadCast(this.broadCast)
        .withOperationEndCallback((): void => this.onDeleteEndBindImpl())
      menuOperation = new ClearRecycleMenuOperation(menuContext);
    } else if (action === Action.RECOVER) {
      if (currentPhoto == undefined) {
        return;
      }
      menuContext = new MenuContext();
      menuContext.withItems([currentPhoto])
        .withOperationEndCallback(this.onDeleteEnd)
        .withBroadCast(this.broadCast)
      menuOperation = new BatchRecoverMenuOperation(menuContext);
    } else if (action === Action.GOTO_PHOTOS) {
      menuContext.withJumpSourceToMain(this.jumpSourceToMain);
      menuOperation = new GotoPhotosMenuOperation(menuContext);
    } else if (action === Action.RENAME) {
      if (currentPhoto == undefined) {
        return;
      }
      menuContext.withItems([currentPhoto])
        .withBroadCast(this.broadCast)
        .withAlbumInfo(this.albumInfo);
      menuOperation = new RenameMenuOperation(menuContext);
    } else if (action === Action.MOVE) {
      this.backFromCopy = true;
      this.routeSelectAlbumPage(MediaOperationType.Move);
      return;
    } else {
      return;
    }
    menuOperation.doAction();
  }

  private setFavourite(isSuccess: boolean): void {
    Log.debug(TAG, 'set favor !');

    if (isSuccess) {
      let currentPhoto = this.getCurrentPhoto();
      if (currentPhoto == undefined) {
        return;
      }
      if (userFileDataItemCache.hasKey(currentPhoto.uri)) {
        userFileDataItemCache.get(currentPhoto.uri).favouriteStatus = currentPhoto.favouriteStatus;
      }
      // 收藏相册进入大图，点击取消收藏会导致Item成员变少，所以要dataRemove
      this.browserDataSource.reloadGroupItemData(false).then<void, void>((): void => {
        this.operateEnd();
      });
      this.updateMenu();
      this.appBroadcast.emit(BroadcastConstants.UPDATE_DATA_SOURCE, [currentPhoto]);
    }
  }

  private operateEnd(): void {
    if (this.currentIndex === 0 || AppStorage.Get<number>('isLeftSwiper') === 0) {
      this.broadCast.emit(BroadcastConstants.ON_DATA_RELOADED, []);
    } else {
      Log.info(TAG, 'operateEnd curIndex : ' + this.currentIndex);
      this.broadCast.emit(BroadcastConstants.ON_DATA_RELOADED, []);
      this.currentIndex--;
    }
  }

  private onDeleteEnd(): void {
    this.onDeleteEndBindImpl()
  }

  private onDeleteEndBindImpl(): void {
    Log.debug(TAG, 'onDeleteEnd start');
    AppStorage.SetOrCreate<number>('isDelete', 0);
    this.operateEnd();
  }

  private setOrientation(): void {
    Log.debug(TAG, 'setOrientation');
    let currentPhoto = this.getCurrentPhoto();
    if (currentPhoto == undefined) {
      return;
    }
    this.browserDataSource.notifyDataChange(this.currentIndex);
    this.appBroadcast.emit(BroadcastConstants.UPDATE_DATA_SOURCE, [currentPhoto]);
  }

  onToggleBars(): void {
    if (this.isShowBar) {
      this.hideBars();
    } else {
      this.showBars();
    }
    Log.info(TAG, 'Toggle bars, isShowBar: ' + this.isShowBar);
  }

  showBars(): void {
    if (!this.isShowBar) {
      this.isShowBar = true;
      this.browserBackgroundColor = $r('app.color.default_background_color');
      screenManager.setSystemUi(true);
    }
  }

  hideBars(): void {
    if (this.isShowBar) {
      this.isShowBar = false;
      this.browserBackgroundColor = $r('app.color.black');
      screenManager.setSystemUi(false);
    }
  }

  routeSelectAlbumPage(pageType: string): void {
    router.push({
      uri: 'pages/MediaOperationPage',
      params: {
        pageType: pageType,
        albumInfo: JSON.stringify(this.albumInfo)
      }
    });
  }

  onPhotoChanged(index: number): void {
    this.onPhotoChangedBindImpl(index)
  }

  private onPhotoChangedBindImpl(index: number): void {
    index = Math.min(index, this.browserDataSource.totalCount() - 1);
    Log.info(TAG, 'onPhotoChanged start ' + index);
    this.currentIndex = this.isToEdit ? 0 : index;
    this.updateActionBar();
  }

  onDataSizeChanged(size: number): void {
    Log.info(TAG, 'onDataSizeChanged, size is ' + size);
    if (size === 0) {
      if (this.pageFrom === Constants.ENTRY_FROM.CARD) {
        let menuContext: MenuContext = new MenuContext();
        menuContext.withJumpSourceToMain(this.jumpSourceToMain);
        let menuOperation = new GotoPhotosMenuOperation(menuContext);
        menuOperation.doAction();
      } else {
        this.onBackPress();
      }
    }
  }

  updateActionBar(): void {
    let currentPhoto = this.getCurrentPhoto();
    if (currentPhoto == undefined || this.isFromViewData) {
      Log.warn(TAG, 'currentPhoto is undefined: ' + (currentPhoto == undefined) + ' this.isFromViewData: ' + this.isFromViewData)
      return;
    }

    // PhotoItems 和 这里都会有load, 极端情况下可能产生一次无效的load
    currentPhoto.load(false).then<void, void>((): void => {
      this.photoDate = DateUtil.getLocalizedDate(currentPhoto.dateAdded);
      this.timeAndLocation = DateUtil.getLocalizedTime(currentPhoto.dateAdded);
      this.updateMenu();
    })
  }

  updateMenu(): void {
    let currentPhoto = this.getCurrentPhoto();
    if (currentPhoto == undefined) {
      return;
    }
    this.toolMenuList = [];
    this.getActionList(currentPhoto).then<void, void>((list: Action[]): void => {
      if (this.pageFrom === Constants.ENTRY_FROM.CAMERA || this.pageFrom === Constants.ENTRY_FROM.CARD) {
        this.menuList = [Action.GOTO_PHOTOS, Action.INFO];
      } else if (this.pageFrom === Constants.ENTRY_FROM.RECYCLE) {
        this.menuList = [];
      } else {
        this.menuList = [Action.INFO];
      }
      if (this.isHorizontal) {
        this.menuList = this.menuList.concat(list);
      } else {
        this.toolMenuList = list;
      }
      this.updateMoreMenu();
    })
  }

  private async getActionList(currentPhoto: UserFileDataItem): Promise<Action[]> {
    Log.info(TAG,'this.pageFrom' + this.pageFrom)
    let list: Action[] = [];
    if (this.pageFrom === Constants.ENTRY_FROM.RECYCLE) {
      list.push(Action.RECOVER, Action.DELETE);
    } else {
      let isFavor = await currentPhoto.isFavor();
      Log.info(TAG,'isFavor' + isFavor)
      list.push(
        isFavor ? Action.FAVORITE : Action.NOT_FAVORITE,
        Action.RENAME,
        Action.DELETE,);
    }
    return list;
  }
  private currentShowChange(): void {
    this.updateMoreMenu();
  }

  private updateMoreMenu(): void {
    let currentPhoto = this.getCurrentPhoto();
    if (currentPhoto == undefined) {
      return;
    }
    Log.info(TAG,'updateMenu')
  }
  getCurrentPhoto(): UserFileDataItem {
    return this.browserDataSource.getDataByIndex(this.currentIndex);
  }

  getPhotoByIndex(index: number): UserFileDataItem {
    return this.browserDataSource.getDataByIndex(index);
  }

  async onMoveEnd(err: object, count: number, total: number): Promise<void> {
    return await this.onMoveEndBindImpl(err, count, total)
  }

  async onMoveEndBindImpl(err: object, count: number, total: number): Promise<void> {
    Log.debug(TAG, 'onMoveEnd count: ' + count + ', total: ' + total);
   