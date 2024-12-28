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
import { Log } from '../utils/Log';
import { MediaConstants } from '../constants/MediaConstants';
import { ToolBar } from '../components/ToolBar';
import { ImageGridItemComponent } from '../components/ImageGridItemComponent';
import { Action } from '../models/Action';
import { ActionBar } from '../components/ActionBar';
import { ActionBarProp } from '../common/ActionBarProp';
import { CustomDialogView } from '../components/CustomDialogView';
import { Broadcast } from '../common/Broadcast';
import { BroadcastConstants } from '../constants/BroadcastConstants';
import { Constants } from '../constants/Constants';
import { MenuOperation } from '../menus/MenuOperation';
import { MenuContext } from '../menus/MenuContext';
import { BatchDeleteMenuOperation } from '../menus/BatchDeleteMenuOperation';
import { BatchRemoveMenuOperation } from '../menus/BatchRemoveMenuOperation';
import { screenManager } from '../common/ScreenManager';
import { broadcastManager } from '../common/BroadcastManager';
import { JumpSourceToMain } from '../models/JumpSourceToMain';
import { getResourceString } from '../utils/ResourceUtils';
import { MediaOperationType } from '../models/MediaOperationType';
import { MoveMenuOperation } from '../menus/MoveMenuOperation';
import { GridScrollBar } from '../components/GridScrollBar';
import { BatchRecoverMenuOperation } from '../menus/BatchRecoverMenuOperation';
import { ClearRecycleMenuOperation } from '../menus/ClearRecycleMenuOperation';
import { SimpleAlbumDataItem } from '../common/SimpleAlbumDataItem';
import { userFileObserver } from '../base/UserFileObserver';
import { UserFileDataChangeCallback } from '../base/UserFileDataChangeCallback';
import { CommonObserverCallback } from '../common/CommonObserverCallback';
import { NoPhotoComponent } from '../components/NoPhotoComponent';
import { PhotoGridBarModel } from '../models/PhotoGridBarModel';
import { GroupItemDataSource } from '../common/GroupItemDataSource';
import { UserFileDataItem } from '../base/UserFileDataItem';
import { showToast } from '../utils/UiUtil';
import { LazyItem } from '../common/ItemDataSource';

const TAG = 'PhotoGridPage';
// Album Page
@Entry
@Component
struct PhotoGridPage {
  @State isSelectUpperLimited: boolean = false;
  @State isHideScrollBar: boolean = true;
  @Provide @Watch('updateActionBar') isEmpty: boolean = false;
  @State gridRowCount: number = 0;
  @Provide @Watch('updateActionBar') isSelectedMode: boolean = false;
  @Provide @Watch('updateActionBar') isAllSelected: boolean = false;
  @Provide('selectedCount') @Watch('updateActionBar') totalSelectedCount: number = 0;
  @StorageLink('isHorizontal') isHorizontal: boolean = screenManager.isHorizontal();
  @Provide broadCast: Broadcast = new Broadcast();
  @Provide isShow: boolean = true;
  @Provide isShowBar: boolean = true;
  @Provide moreMenuList: Action[] = [];
  @StorageLink('PhotoGridPageIndex') @Watch('onIndexChange') PhotoGridPageIndex: number = Constants.INVALID;
  @StorageLink('isSplitMode') isSplitMode: boolean = screenManager.isSplitMode();
  @StorageLink('leftBlank') leftBlank: number[] = [0, 0, 0, 0];
  private albumInfo: SimpleAlbumDataItem = new SimpleAlbumDataItem('', '', '', '', '', 0, 0);
  title: string = '';
  deviceName: string = '';
  scroller: Scroller | null = new Scroller();
  isDataFreeze = false;
  isActive = false;
  deleteMode: boolean = false;
  deleteReady = true;
  routerStart = false;
  private dataObserver: CommonObserverCallback | null = new CommonObserverCallback(this as UserFileDataChangeCallback);
  private appBroadcast: Broadcast = broadcastManager.getBroadcast();
  private barModel: PhotoGridBarModel | null = new PhotoGridBarModel();
  private groupDataSource: GroupItemDataSource = new GroupItemDataSource();
  private isMediaLibDataChanged: boolean = true;
  @State actionBarProp: ActionBarProp = new ActionBarProp();
  @State toolBarMenuList: Action[] = [];

  updateActionBar(): void {
    if (this.barModel !== undefined && this.barModel !== null) {
      this.actionBarProp = this.barModel.createActionBar(this.title, this.isSelectedMode,
        this.totalSelectedCount, this.isAllSelected, this.isEmpty);
      this.toolBarMenuList = this.barModel.getMenuList(this.isSelectedMode, this.totalSelectedCount,
        this.isAllSelected, this.isEmpty);
    }

  }

  onIndexChange(): void {
    Log.info(TAG, 'onIndexChange ' + this.PhotoGridPageIndex)
    if (this.PhotoGridPageIndex !== Constants.INVALID) {
      this.scroller?.scrollToIndex(this.PhotoGridPageIndex);
    }
  }

  onMenuClicked(action: Action): void {
    this.onMenuClickedBindImpl(action)
  }

  private onMenuClickedBindImpl(action: Action): void {
    Log.info(TAG, 'onMenuClicked, action: ' + action.actionID);
    let menuContext: MenuContext;
    let menuOperation: MenuOperation;
    if (action === Action.BACK) {
      if (AppStorage.Get<boolean>(Constants.BACK_FROM_FORM_DETAIL) != null) {
        this.routerAlbumSetPage();
      } else {
        router.back();
      }
    } else if (action === Action.CANCEL) {
      this.onModeChange();
    } else if (action === Action.SELECT_ALL || action === Action.DESELECT_ALL) {
      this.isAllSelected = !this.groupDataSource.isSelect()
      this.groupDataSource.setSelect(this.isAllSelected)
      this.onSelect()
    } else if (action === Action.DELETE) {
      menuContext = new MenuContext();
      menuContext
        .withDeletePageFromType(BroadcastConstants.DELETE_FROM_GRID)
        .withAlbumId(this.albumInfo.id)
        .withDataSource(this.groupDataSource)
        .withOperationStartCallback((): void => this.onDeleteStart())
        .withOperationEndCallback((err: object): void => this.onDeleteEnd(err))
        .withBroadCast(this.broadCast)
        .withAlbumInfo(this.albumInfo)
      menuOperation = new BatchDeleteMenuOperation(menuContext);
      menuOperation.doAction();
    } else if (action === Action.INFO) {
      this.openDetailsDialog();
    } else if (action === Action.CLEAR_RECYCLE) {
      menuContext = new MenuContext();
      menuContext
        .withDataSource(this.groupDataSource)
        .withOperationStartCallback((): void => this.onDeleteStart())
        .withOperationEndCallback((err: object): void => this.onDeleteEnd(err))
        .withBroadCast(this.broadCast)
      menuOperation = new ClearRecycleMenuOperation(menuContext)
      menuOperation.doAction();
    } else if (action === Action.RECOVER) {
      menuContext = new MenuContext();
      menuContext
        .withDataSource(this.groupDataSource)
        .withOperationStartCallback((): void => this.onDeleteStart())
        .withOperationEndCallback((err: object): void => this.onDeleteEnd(err))
        .withBroadCast(this.broadCast)
      menuOperation = new BatchRecoverMenuOperation(menuContext);
      menuOperation.doAction();
    } else if (action === Action.MOVE) {
      this.routeSelectAlbumPage(MediaOperationType.Move);
    } else if (action === Action.REMOVE) {
      menuContext = new MenuContext();
      menuContext
        .withAlbumId(this.albumInfo.id)
        .withDataSource(this.groupDataSource)
        .withOperationStartCallback((): void => this.onDeleteStart())
        .withOperationEndCallback((err: object): void => this.onDeleteEnd(err))
        .withBroadCast(this.broadCast)
        .withAlbumInfo(this.albumInfo)
      menuOperation = new BatchRemoveMenuOperation(menuContext);
      menuOperation.doAction();
    } else if (action === Action.NEW) {
      this.routeToAddMediaPage();
    }
  }

  async openDetailsDialog(): Promise<void> {
    let items: UserFileDataItem[] = this.groupDataSource.getSelectedItems() as UserFileDataItem[];
    if (items.length === 0) {
      Log.warn(TAG, 'no select error');
    } else if (items.length === 1) {
      Log.info(TAG, 'totalSelectedCount is 1');
      this.broadCast.emit(BroadcastConstants.SHOW_DETAIL_DIALOG, [items[0]]);
    } else {
      let size = 0;
      items.forEach((item: UserFileDataItem): void => {
        Log.debug(TAG, 'openDetailsDialog item.size: ' + item.size);
        size = size + item.size;
      })
      Log.info(TAG, 'openDetailsDialog size: ' + size);
      this.broadCast.emit(BroadcastConstants.SHOW_MULTI_SELECT_DIALOG, [items.length, size]);
    }
  }
  private routeSelectAlbumPage(pageType: string): void {
    this.routerStart = true;
    router.push({
      uri: 'pages/MediaOperationPage',
      params: {
        pageType: pageType,
        albumInfo: JSON.stringify(this.albumInfo)
      }
    });
  }
  private routerAlbumSetPage(): void {
    router.replace({
      uri: 'pages/index',
      params: {
        jumpSource: JumpSourceToMain.ALBUM,
      }
    });
  }

  private routeToAddMediaPage(): void {
    router.push({
      uri: 'pages/AlbumSelect',
      params: {
        albumInfo: JSON.stringify(this.albumInfo)
      }
    });
  }

  onCopyStart(): void {
    this.onCopyStartBindImpl()
  }

  private onCopyStartBindImpl(): void {
    Log.info(TAG, 'onCopyStart');
    userFileObserver.unregisterObserver(this.dataObserver);
  }

  onCopyEnd(err: object, count: number, total: number): void {
    this.onCopyEndBindImpl(err, count, total)
  }

  onCopyEndBindImpl(err: object, count: number, total: number): void {
    Log.info(TAG, 'onCopyEnd count: ' + count + ', total: ' + total);
    this.onModeChange();
    userFileObserver.registerObserver(this.dataObserver);
    if (new Boolean(err).valueOf()) {
      getResourceString($r('app.string.copy_failed_single')).then<void, void>((message: string): void => {
        showToast(message);
      })
    } else {
      animateTo({ duration: Constants.DELETE_ANIMATE_DURATION }, (): void => this.groupDataSource.dataReload(true));
      this.isHideScrollBar = (this.groupDataSource.totalCount() <= Constants.GRID_SCROLL_BAR_VISIBLE_THRESHOLD)
    }
  }

  onDownloadStart(): void {
    this.onDownloadStartBindImpl()
  }

  private onDownloadStartBindImpl(): void {
    Log.info(TAG, 'onDownloadStart');
    userFileObserver.unregisterObserver(this.dataObserver);
  }

  async onDownloadEnd(err: object, count: number, total: number): Promise<void> {
    return await this.onDownloadEndBindImpl(err, count, total)
  }

  async onDownloadEndBindImpl(err: object, count: number, total: number): Promise<void> {
    Log.info(TAG, 'onDownloadEnd count: ' + count + ', total: ' + total);
    this.onModeChange();
    userFileObserver.registerObserver(this.dataObserver);
    if (new Boolean(err).valueOf()) {
      if (total > 1) {
        Log.warn(TAG, 'get selectItems is error ' + count);
        let str = await getResourceString($r('app.string.download_failed_multi'));
        let message = str.replace('%d', count.toString());
        showToast(message);
      } else {
        getResourceString($r('app.string.download_failed_single')).then<void, void>((message: string): void => {
          showToast(message);
        })
      }
      ;
    } else {
      getResourceString($r('app.string.download_progress_done')).then<void, void>((message: string): void => {
        showToast(message);
      })
    }
  }

  onMoveStart(): void {
    this.onMoveStartBindImpl()
  }

  private onMoveStartBindImpl(): void {
    Log.info(TAG, 'onMoveStart');
    userFileObserver.unregisterObserver(this.dataObserver);
  }

  onMoveEnd(err: object, count: number, total: number): void {
    this.onMoveEndBindImpl(err, count, total)
  }

  private onMoveEndBindImpl(err: object, count: number, total: number): void {
    Log.info(TAG, 'onMoveEnd count: ' + count + ', total: ' + total);
    this.onModeChange();
    userFileObserver.registerObserver(this.dataObserver);
    if (new Boolean(err).valueOf()) {
      getResourceString($r('app.string.move_failed_single')).then<void, void>((message: string): void => {
        showToast(message);
      })
    } else {
      animateTo({ duration: Constants.DELETE_ANIMATE_DURATION }, (): void => this.groupDataSource.dataReload(true));
    }
  }

  onDeleteStart(): void {
    this.onDeleteStartBindImpl()
  }

  private onDeleteStartBindImpl(): void {
    Log.info(TAG, 'onDeleteStart');
    userFileObserver.unregisterObserver(this.dataObserver);
  }

  onDeleteEnd(err: object): void {
    this.onDeleteEndBindImpl(err)
  }

  private onDeleteEndBindImpl(err: object): void {
    Log.info(TAG, 'onDeleteEnd');
    this.onModeChange();
    userFileObserver.registerObserver(this.dataObserver);

    animateTo({ duration: Constants.DELETE_ANIMATE_DURATION }, (): void => {
      this.groupDataSource.dataRemove();
      if (this.groupDataSource.groupDataItem) {
        this.isEmpty = this.groupDataSource.groupDataItem.length == 0 ? true : false;
      }
      this.isHideScrollBar = (this.groupDataSource.totalCount() <= Constants.GRID_SCROLL_BAR_VISIBLE_THRESHOLD)
    })
    //for NAPI notify delete bug to load manually
    this.onUserFileDataChange(MediaConstants.MEDIA_TYPE_IMAGE);
  }

  /**
   * Just change select mode to false
   */
  onModeChange(): void {
    Log.info(TAG, 'onModeChange');
    if (this.isSelectedMode) {
      this.isSelectedMode = false;
      this.groupDataSource.setSelect(false);
      this.onSelect();
    }
  }

  onPageShow(): void {
    Log.error(TAG,"onPageShow" + this.isActive)
    this.appBroadcast.emit(BroadcastConstants.THIRD_ROUTE_PAGE, []);
    this.isShow = true;
    let params: ParamsInterface = router.getParams();
    if (this.routerStart && params != null
      && params.pageType != null
      && params.albumInfo != null) {
      Log.info(TAG, 'MediaOperation back');
      if (params.pageType === MediaOperationType.Move) {
        this.moveOperation(JSON.parse(params.albumInfo.toString()));
      }
    }
    this.routerStart =