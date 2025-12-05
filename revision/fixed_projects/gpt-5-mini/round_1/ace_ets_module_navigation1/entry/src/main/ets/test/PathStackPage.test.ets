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

import CommonFunc from '../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function PathStackPage() {
  describe('PathStackPage', () => {
    beforeEach(async (done: Function) => {
      console.info('PathStackPage beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/PathStackPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get PathStackPage state pages:' + JSON.stringify(pages));
        if (!('PathStackPage' == pages.name)) {
          console.info('get PathStackPage pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push PathStackPage page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push PathStackPage page error:' + err);
      }
      console.info('PathStackPage beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('PathStackPage after each called');
    });


    /**
     * @tc.number    PathStackPage_0100
     * @tc.name      PathStackPage_0100
     * @tc.desc      Test navDestinationId、setPathStack、getPathStack
     * @tc.level     Level 2
     */
    it('PathStackPage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('PathStackPage_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(500);

      let setBtn: Component = await driver.findComponent(ON.id('pushId'));
      await setBtn.click()
      console.info('[PathStackPage_0100] setBtn click');
      await driver.delayMs(1000);
      
      let getBtn: Component = await driver.findComponent(ON.id('getId'));
      await getBtn.click()
      console.info('[PathStackPage_0100] getBtn click');
      await driver.delayMs(2000);

      let stackLen: number = AppStorage.get('stackLen') as number
      let navDestinationId1: number = AppStorage.get('navDestinationId1') as number
      expect(stackLen).assertEqual(2);
      expect(navDestinationId1).assertEqual(0);
      AppStorage.delete('stackLen');
      AppStorage.delete('navDestinationId1')

      console.info('PathStackPage_0100 END');
      done();
    });

  })
}