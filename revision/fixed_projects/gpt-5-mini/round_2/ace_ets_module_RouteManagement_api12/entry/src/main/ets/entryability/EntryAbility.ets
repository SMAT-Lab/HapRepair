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
import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { BusinessError } from '@ohos.base';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam){
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onCreate');
    globalThis.abilityWant = want;
  }

  onDestroy(){
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onDestroy');
  }
  
  onWindowStageCreate(windowStage: window.WindowStage):void{
    // Main window is created, set main page for this ability
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageCreate');

    globalThis.windowStage = windowStage
    globalThis.abilityContext = this.context
    windowStage.loadContent('MainAbility/pages/index/index', (err: BusinessError, data) => {
      if (err.code) {
        hilog.error(0x0000, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err) ?? '');
        return;
      }
      hilog.info(0x0000, 'testTag', 'Succeeded in loading the content.');
      let windowClass: window.Window | undefined = undefined;
      windowStage.getMainWindow((err: BusinessError, data) => {
        let errCode: number = err.code;
        if (errCode) {
          console.error('Failed to obtain the main window. Cause: ' + JSON.stringify(err));
          return;
        }
        windowClass = data;
        console.info('Succeeded in obtaining the main window. Data: ' + JSON.stringify(data));
        try {
          let properties: window.WindowProperties = windowClass.getWindowProperties();
          let wRect: window.Rect =  properties.windowRect;
          globalThis.winLeft = wRect.left;
          globalThis.winTop = wRect.top;
          console.info('Succeeded get winLeft:' + globalThis.winLeft + ',winTop:' + globalThis.winTop );
        } catch (exception) {
          console.error('Failed to obtain the window properties. Cause: ' + JSON.stringify(exception));
        }
      })
    });
  }

  onWindowStageDestroy():void{
    // Main window is destroyed, release UI related resources
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground():void{
    // Ability has brought to foreground
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onForeground');
  }

  onBackground():void{
    // Ability has back to background
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onBackground');
  }
}