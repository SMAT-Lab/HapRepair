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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function EnableModeChangeAnimation() {
  describe('EnableModeChangeAnimation', () => {
    beforeEach(async (done: Function) => {
      console.info('EnableModeChangeAnimation beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/EnableModeChangeAnimation',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get EnableModeChangeAnimation state pages:' + JSON.stringify(pages));
        if (!('EnableModeChangeAnimation' == pages.name)) {
          console.info('get EnableModeChangeAnimation pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(1000);
          console.info('push EnableModeChangeAnimation page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push EnableModeChangeAnimation page error:' + err);
      }
      console.info('EnableModeChangeAnimation beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('EnableModeChangeAnimation after each called');
    });


    /**
     * @tc.number    EnableModeChangeAnimation_0100
     * @tc.name      EnableModeChangeAnimation_0100
     * @tc.desc      Test EnableModeChangeAnimation true
     * @tc.level     Level 2
     */
    it('EnableModeChangeAnimation_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('EnableModeChangeAnimation_0100 START');
      await CommonFunc.sleep(200);

      let driver = Driver.create();
      await CommonFunc.sleep(500);

      let enableTrueId = await driver.findComponent(ON.id('enableTrueId'));
      await enableTrueId.click()
      console.info('[EnableModeChangeAnimation_0100] enableTrueId click');
      await CommonFunc.sleep(200);
	  
      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let splitId = await scrollBar.scrollSearch(ON.id('splitId'));
      await splitId.click()
      console.info('[EnableModeChangeAnimation_0100] splitId click');
      await CommonFunc.sleep(200);

      let stackId = await scrollBar.scrollSearch(ON.id('stackId'));
      await stackId.click()
      console.info('[EnableModeChangeAnimation_0100] stackId click');
      await CommonFunc.sleep(200);

      let strJson: string = getInspectorByKey('navigationId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[EnableModeChangeAnimation_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[EnableModeChangeAnimation_0100] enableModeChangeAnimation obj is: ' + JSON.stringify(obj.$attrs.enableModeChangeAnimation));
      expect(obj.$attrs.enableModeChangeAnimation).assertTrue();

      console.info('EnableModeChangeAnimation_0100 END');
      done();
    });

        /**
     * @tc.number    EnableModeChangeAnimation_0200
     * @tc.name      EnableModeChangeAnimation_0200
     * @tc.desc      Test EnableModeChangeAnimation false
     * @tc.level     Level 2
     */
    it('EnableModeChangeAnimation_0200',  Level.LEVEL0, async (done: Function) => {
      console.info('EnableModeChangeAnimation_0200 START');
      await CommonFunc.sleep(200);

      let driver = Driver.create();
      await CommonFunc.sleep(500);
	  
      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let enableFalseId = await scrollBar.scrollSearch(ON.id('enableFalseId'));
      await enableFalseId.click()
      console.info('[EnableModeChangeAnimation_0200] enableFalseId click');
      await CommonFunc.sleep(200);

      let splitId = await scrollBar.scrollSearch(ON.id('splitId'));
      await splitId.click()
      console.info('[EnableModeChangeAnimation_0200] splitId click');
      await CommonFunc.sleep(200);

      let stackId = await scrollBar.scrollSearch(ON.id('stackId'));
      await stackId.click()
      console.info('[EnableModeChangeAnimation_0200] stackId click');
      await CommonFunc.sleep(200);

      let strJson: string = getInspectorByKey('navigationId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[EnableModeChangeAnimation_0200] component obj is: ' + JSON.stringify(obj));
      console.info('[EnableModeChangeAnimation_0200] enableModeChangeAnimation obj is: ' + JSON.stringify(obj.$attrs.enableModeChangeAnimation));
      expect(obj.$attrs.enableModeChangeAnimation).assertFalse();

      console.info('EnableModeChangeAnimation_0200 END');
      done();
    });

  })
}