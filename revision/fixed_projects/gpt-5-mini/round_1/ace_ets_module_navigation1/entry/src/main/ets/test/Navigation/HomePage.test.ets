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

export default function HomePage() {
  describe('HomePage', () => {
    beforeEach(async (done: Function) => {
      console.info('HomePage beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/HomePage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get HomePage state pages:' + JSON.stringify(pages));
        if (!('HomePage' == pages.name)) {
          console.info('get HomePage pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push HomePage page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push HomePage page error:' + err);
      }
      console.info('HomePage beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('HomePage after each called');
    });


    /**
     * @tc.number    HomePage_0100
     * @tc.name      HomePage_0100
     * @tc.desc      Test HomePageInfo name and param
     * @tc.level     Level 0
     */
    it('HomePage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('HomePage_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let name: string = AppStorage.get('name') as string
      console.log('[HomePage_0100] name value : ' + name)
      expect(name).assertEqual('PageOne');
      AppStorage.delete('name');

      let param: string = AppStorage.get('param') as string
      console.log('[HomePage_0100] param value : ' + param)
      expect(param).assertEqual('{"key":"value"}');
      AppStorage.delete('param');

      console.info('HomePage_0100 END');
      done();
    });

  })
}