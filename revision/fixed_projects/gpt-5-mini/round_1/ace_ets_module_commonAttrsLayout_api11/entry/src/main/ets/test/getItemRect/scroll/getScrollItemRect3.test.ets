/**
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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import {
  UiComponent,
  UiDriver,
  Component,
  Driver,
  UiWindow,
  ON,
  BY,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';
import CommonFunc from '../../../MainAbility/common/Common';
import { MessageManager, Callback } from '../../../MainAbility/common/MessageManager';


export default function getScrollItemRect3() {
  describe('getScrollItemRect3', () => {
    beforeEach(async (done: Function) => {
      console.info("getScrollItemRect3 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/getItemRect/scroll/getScrollItemRect3',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get getScrollItemRect3 state pages: " + JSON.stringify(pages));
        if (!("getScrollItemRect3" == pages.name)) {
          console.info("get getScrollItemRect3 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push getScrollItemRect3 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getScrollItemRect3 page error: " + err);
        expect().assertFail();
      }
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back({
        url: 'MainAbility/pages/index/index'
      })
      console.info("BadgeFocus after each called");
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getScrollItemRect3_0100
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 0,set position of column to {x:20,y:0}
    * @tc.level : Level 2
    */
    it('getScrollItemRect3_0100', Level.LEVEL0, async (done: Function) => {

      console.info('[getScrollItemRect3_0100] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'x1', value: 20
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getScrollItemRect3_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getScrollItemRect3_text1');
      let text2 = CommonFunc.getComponentRect('getScrollItemRect3_text2');
      let text3 = CommonFunc.getComponentRect('getScrollItemRect3_text3');
      let text4 = CommonFunc.getComponentRect('getScrollItemRect3_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getScrollItemRect3_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getScrollItemRect3_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getScrollItemRect3_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getScrollItemRect3_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(0));

      expect(Math.round(y)).assertEqual(Math.round(0));

      console.log('assert size')

      expect(Math.round(width)).assertEqual(Math.round(300));

      expect(Math.round(height)).assertEqual(Math.round(600));

      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getScrollItemRect3_0200
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 0,set position of column to {x:0,y:20}
    * @tc.level : Level 2
    */
    it('getScrollItemRect3_0200', Level.LEVEL0, async (done: Function) => {

      console.info('[getScrollItemRect3_0200] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'y1', value: 20
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getScrollItemRect3_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getScrollItemRect3_text1');
      let text2 = CommonFunc.getComponentRect('getScrollItemRect3_text2');
      let text3 = CommonFunc.getComponentRect('getScrollItemRect3_text3');
      let text4 = CommonFunc.getComponentRect('getScrollItemRect3_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getScrollItemRect3_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getScrollItemRect3_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getScrollItemRect3_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getScrollItemRect3_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(0));
      expect(Math.round(y)).assertEqual(Math.round(0));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(300));
      expect(Math.round(height)).assertEqual(Math.round(600));

      done();
    });


  })
}
