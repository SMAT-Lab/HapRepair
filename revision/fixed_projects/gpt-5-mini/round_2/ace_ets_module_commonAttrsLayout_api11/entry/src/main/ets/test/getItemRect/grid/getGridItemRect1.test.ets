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


export default function getGridItemRect1() {
  describe('getGridItemRect1', () => {
    beforeEach(async (done: Function) => {
      console.info("getGridItemRect1 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/getItemRect/grid/getGridItemRect1',
      }
      try {
        router.clear();
        await CommonFunc.sleep(1000);
        let pages = router.getState();
        console.info("get getGridItemRect1 state pages: " + JSON.stringify(pages));
        if (!("getGridItemRect1" == pages.name)) {
          console.info("get getGridItemRect1 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push getGridItemRect1 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getGridItemRect1 page error: " + err);
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
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect1_0100
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 3
    * @tc.level : Level 2
    */
    it('getGridItemRect1_0100', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect1_0100] START');
      globalThis.value.message.notify({
        name: 'index', value: 3
      });
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let gridItem_3 = CommonFunc.getComponentRect('getGridItemRect1_gridItem_3');

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect1_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect1_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect1_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(240));

      expect(Math.round(y)).assertEqual(Math.round(0));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(60));
      expect(Math.round(height)).assertEqual(Math.round(100));

      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect1_0200
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 4
    * @tc.level : Level 2
    */
    it('getGridItemRect1_0200', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect1_0200] START');
      globalThis.value.message.notify({
        name: 'index', value: 4
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect1_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect1_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect1_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(60));
      expect(Math.round(y)).assertEqual(Math.round(100));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(60));
      expect(Math.round(height)).assertEqual(Math.round(100));

      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect1_0400
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 100
    * @tc.level : Level 2
    */
    it('getGridItemRect1_0400', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect1_0400] START');
      globalThis.value.message.notify({
        name: 'index', value: 100
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect1_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect1_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect1_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(0));
      expect(Math.round(y)).assertEqual(Math.round(0));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(0));
      expect(Math.round(height)).assertEqual(Math.round(0));
      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect1_0500
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 0
    * @tc.level : Level 2
    */
    it('getGridItemRect1_0500', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect1_0500] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect1_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect1_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect1_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(0));
      expect(Math.round(y)).assertEqual(Math.round(0));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(60));
      expect(Math.round(height)).assertEqual(Math.round(200));
      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect1_0600
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 1
    * @tc.level : Level 2
    */
    it('getGridItemRect1_0600', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect1_0600] START');
      globalThis.value.message.notify({
        name: 'index', value: 1
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect1_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect1_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect1_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(60));
      expect(Math.round(y)).assertEqual(Math.round(0));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(120));
      expect(Math.round(height)).assertEqual(Math.round(100));
      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect1_0700
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 0,set the padding of grid to 20
    * @tc.level : Level 2
    */
    it('getGridItemRect1_0700', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect1_0700] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'padding1', value: 20
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect1_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect1_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect1_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(20));
      expect(Math.round(y)).assertEqual(Math.round(20));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(52));
      expect(Math.round(height)).assertEqual(Math.round(184));
      done();
    });

  })
}
