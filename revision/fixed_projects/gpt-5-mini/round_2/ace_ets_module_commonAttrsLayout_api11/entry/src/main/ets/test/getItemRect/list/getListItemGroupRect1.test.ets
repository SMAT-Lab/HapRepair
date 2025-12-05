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


export default function getListItemGroupRect1() {
  describe('getListItemGroupRect1', () => {
    beforeEach(async (done: Function) => {
      console.info("getListItemGroupRect1 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/getItemRect/list/getListItemGroupRect1',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get getListItemGroupRect1 state pages: " + JSON.stringify(pages));
        if (!("getListItemGroupRect1" == pages.name)) {
          console.info("get getListItemGroupRect1 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push getListItemGroupRect1 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getListItemGroupRect1 page error: " + err);
        expect().assertFail();
      }
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(500);
      let btn3 = await driver.findComponent(ON.id('getListItemGroupRect1_button3'));
      await btn3.click();
      await CommonFunc.sleep(3000);
      console.info("BadgeFocus after each called");
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0100
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 0,set indexInGroup to 0
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0100', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0100] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 0
      });
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let listItemGroup_0 = CommonFunc.getComponentRect('getListItemGroupRect1_listItemGroup_0');

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(50)) <= 1).assertTrue();

      console.log('assert size')
      console.log('listItemGroup_0.right - listItemGroup_0.left' + (listItemGroup_0.right - listItemGroup_0.left))
      expect(Math.abs(vp2px(Math.round(width)) - Math.round(listItemGroup_0.right - listItemGroup_0.left)) <= 1).assertTrue();
      console.log('listItemGroup_0.bottom - listItemGroup_0.top' + (listItemGroup_0.bottom - listItemGroup_0.top))
      expect(Math.abs(vp2px(Math.round(height)) - Math.round(listItemGroup_0.bottom - listItemGroup_0.top)) <= 1).assertTrue();

      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0200
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 0,set indexInGroup to 1
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0200', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0200] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 1
      });
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let listItemGroup_1 = CommonFunc.getComponentRect('getListItemGroupRect1_listItemGroup_1');

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(60)) <= 1).assertTrue();

      console.log('assert size')
      console.log('listItemGroup_1.right - listItemGroup_1.left' + (listItemGroup_1.right - listItemGroup_1.left))
      expect(Math.abs(vp2px(Math.round(width)) - Math.round(listItemGroup_1.right - listItemGroup_1.left)) <= 1).assertTrue();
      console.log('listItemGroup_1.bottom - listItemGroup_1.top' + (listItemGroup_1.bottom - listItemGroup_1.top))
      expect(Math.abs(vp2px(Math.round(height)) - Math.round(listItemGroup_1.bottom - listItemGroup_1.top)) <= 1).assertTrue();

      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0300
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 0,set indexInGroup to -1
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0300', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0300] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: -1
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(0)) <= 1).assertTrue();

      console.log('assert size')
      expect(Math.abs(Math.round(width) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(height) - Math.round(0)) <= 1).assertTrue();

      done();
    });
    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0400
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 0,set indexInGroup to 0,The first sub component slides out of the screen
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0400', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0300] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 0
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button2 click')
      let btn2 = await driver.findComponent(ON.id('getListItemGroupRect1_button2'));
      await btn2.click();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(50)) <= 1).assertTrue();

      console.log('assert size')
      expect(Math.abs(Math.round(width) - Math.round(300)) <= 1).assertTrue();
      expect(Math.abs(Math.round(height) - Math.round(10)) <= 1).assertTrue();

      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0500
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 0,set indexInGroup to 100
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0500', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0500] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 100
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(0)) <= 1).assertTrue();

      console.log('assert size')
      expect(Math.abs(Math.round(width) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(height) - Math.round(0)) <= 1).assertTrue();

      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0600
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 1,set indexInGroup to 0
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0600', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0600] START');
      globalThis.value.message.notify({
        name: 'index', value: 1
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 0
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(0)) <= 1).assertTrue();

      console.log('assert size')
      expect(Math.abs(Math.round(width) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(height) - Math.round(0)) <= 1).assertTrue();

      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0700
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 1,set indexInGroup to 0,the space of lisItemGroup set to 20
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0700', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0700] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 1
      });
      globalThis.value.message.notify({
        name: 'space2', value: 20
      });
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let listItemGroup_1 = CommonFunc.getComponentRect('getListItemGroupRect1_listItemGroup_1');

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(0)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(80)) <= 1).assertTrue();

      console.log('assert size')
      console.log('listItemGroup_1.right - listItemGroup_1.left' + (listItemGroup_1.right - listItemGroup_1.left))
      expect(Math.abs(vp2px(Math.round(width)) - Math.round(listItemGroup_1.right - listItemGroup_1.left)) <= 1).assertTrue();
      console.log('listItemGroup_1.bottom - listItemGroup_1.top' + (listItemGroup_1.bottom - listItemGroup_1.top))
      expect(Math.abs(vp2px(Math.round(height)) - Math.round(listItemGroup_1.bottom - listItemGroup_1.top)) <= 1).assertTrue();

      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0800
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 1,set indexInGroup to 0,the padding of lisItemGroup set to 20
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0800', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0800] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 0
      });
      globalThis.value.message.notify({
        name: 'padding1', value: 20
      });
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let listItemGroup_0 = CommonFunc.getComponentRect('getListItemGroupRect1_listItemGroup_0');

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.abs(Math.round(x) - Math.round(20)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(70)) <= 1).assertTrue();

      console.log('assert size')
      console.log('listItemGroup_0.right - listItemGroup_0.left' + (listItemGroup_0.right - listItemGroup_0.left))
      expect(Math.floor(Math.abs(vp2px(width) - (listItemGroup_0.right - listItemGroup_0.left))) <= 1).assertTrue();
      console.log('listItemGroup_0.bottom - listItemGroup_0.top' + (listItemGroup_0.bottom - listItemGroup_0.top))
      expect(Math.floor(Math.abs(vp2px(height) - (listItemGroup_0.bottom - listItemGroup_0.top))) <= 1).assertTrue();


      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemGroupRect1_0900
    * @tc.name  : testGetItemRectInGroup
    * @tc.desc  : set index to 1,set indexInGroup to 0,the margin of lisItem set to 20
    * @tc.level : Level 2
    */
    it('getListItemGroupRect1_0900', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemGroupRect1_0900] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'indexInGroup', value: 0
      });
      globalThis.value.message.notify({
        name: 'padding1', value: 0
      });
      globalThis.value.message.notify({
        name: 'margin1', value: 20
      });
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let listItemGroup_0 = CommonFunc.getComponentRect('getListItemGroupRect1_listItemGroup_0');

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemGroupRect1_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemGroupRect1_text1');
      let text2 = CommonFunc.getComponentRect('getListItemGroupRect1_text2');
      let text3 = CommonFunc.getComponentRect('getListItemGroupRect1_text3');
      let text4 = CommonFunc.getComponentRect('getListItemGroupRect1_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemGroupRect1_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemGroupRect1_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemGroupRect1_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemGroupRect1_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')

      expect(Math.abs(Math.round(x) - Math.round(20)) <= 1).assertTrue();
      expect(Math.abs(Math.round(y) - Math.round(70)) <= 1).assertTrue();

      console.log('assert size')
      console.log('listItemGroup_0.right - listItemGroup_0.left' + (listItemGroup_0.right - listItemGroup_0.left))
      expect(Math.abs(vp2px(Math.round(width)) - Math.round(listItemGroup_0.right - listItemGroup_0.left)) <= 1).assertTrue();
      console.log('listItemGroup_0.bottom - listItemGroup_0.top' + (listItemGroup_0.bottom - listItemGroup_0.top))
      expect(Math.abs(vp2px(Math.round(height)) - Math.round(listItemGroup_0.bottom - listItemGroup_0.top)) <= 1).assertTrue();

      done();
    });


  })
}
