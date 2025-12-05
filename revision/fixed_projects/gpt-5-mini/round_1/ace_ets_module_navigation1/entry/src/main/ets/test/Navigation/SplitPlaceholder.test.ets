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

export default function SplitPlaceholder() {
  describe('SplitPlaceholder', () => {
    beforeEach(async (done: Function) => {
      console.info('SplitPlaceholder beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/SplitPlaceholder',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get SplitPlaceholder state pages:' + JSON.stringify(pages));
        if (!('SplitPlaceholder' == pages.name)) {
          console.info('get SplitPlaceholder pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push SplitPlaceholder page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push SplitPlaceholder page error:' + err);
      }
      console.info('SplitPlaceholder beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('SplitPlaceholder after each called');
    });


    /**
     * @tc.number    SplitPlaceholder_0100
     * @tc.name      SplitPlaceholder_0100
     * @tc.desc      Test  SplitPlaceholder exist
     * @tc.level     Level 0
     */
    it('SplitPlaceholder_0100', Level.LEVEL0, async (done: Function) => {
      console.info('[SplitPlaceholder_0100] START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let obj1: ESObject = JSON.parse(JSON.stringify(getInspectorTree()));
      await CommonFunc.sleep(1000);

      console.log('SplitPlaceholder_0100: ', JSON.stringify(obj1.$children[0].$children[3]))
      console.log('SplitPlaceholder_0100]: ', obj1.$children[0].$children[3].$type)

      await driver.assertComponentExist(ON.type('SplitPlaceholderContentNode'));
      let splitPlaceholderContentNode: Component | null = await driver.findComponent(ON.type('SplitPlaceholderContentNode'));
      expect(splitPlaceholderContentNode !== null).assertTrue();

      console.info('[SplitPlaceholder_0100] END');
      done();
    });

    /**
     * @tc.number    SplitPlaceholder_0200
     * @tc.name      SplitPlaceholder_0200
     * @tc.desc      Test set SplitPlaceholder null
     * @tc.level     Level 0
     */
    it('SplitPlaceholder_0200', Level.LEVEL0, async (done: Function) => {
      console.info('[SplitPlaceholder_0200] START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn4: Component = await driver.findComponent(ON.id('btn4'));
      await btn4.click()
      console.info('[SplitPlaceholder_0200] btn4 click');
      await driver.delayMs(1000);

      let splitPlaceholderContentNode: Component | null = await driver.findComponent(ON.type('SplitPlaceholderContentNode'));
      expect(splitPlaceholderContentNode).assertEqual(null);

      console.info('[SplitPlaceholder_0200] END');
      done();
    });

    /**
     * @tc.number    SplitPlaceholder_0300
     * @tc.name      SplitPlaceholder_0300
     * @tc.desc      Test set SplitPlaceholder undefined
     * @tc.level     Level 0
     */
    it('SplitPlaceholder_0300', Level.LEVEL0, async (done: Function) => {
      console.info('[SplitPlaceholder_0300] START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn5: Component = await driver.findComponent(ON.id('btn5'));
      await btn5.click()
      console.info('[SplitPlaceholder_0300] btn5 click');
      await driver.delayMs(1000);

      let splitPlaceholderContentNode: Component | null = await driver.findComponent(ON.type('SplitPlaceholderContentNode'));
      expect(splitPlaceholderContentNode).assertEqual(null);
      console.info('[SplitPlaceholder_0300] END');
      done();
    });

    /**
     * @tc.number    SplitPlaceholder_0400
     * @tc.name      SplitPlaceholder_0400
     * @tc.desc      Test set SplitPlaceholder 空节点
     * @tc.level     Level 0
     */
    it('SplitPlaceholder_0400', Level.LEVEL0, async (done: Function) => {
      console.info('[SplitPlaceholder_0400] START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn3: Component = await driver.findComponent(ON.id('btn3'));
      await btn3.click()
      console.info('[SplitPlaceholder_0400] btn3 click');
      await driver.delayMs(1000);

      let splitPlaceholderContentNode: Component | null = await driver.findComponent(ON.type('SplitPlaceholderContentNode'));
      expect(splitPlaceholderContentNode).assertEqual(null);
      console.info('[SplitPlaceholder_0400] END');
      done();
    });

  })
}