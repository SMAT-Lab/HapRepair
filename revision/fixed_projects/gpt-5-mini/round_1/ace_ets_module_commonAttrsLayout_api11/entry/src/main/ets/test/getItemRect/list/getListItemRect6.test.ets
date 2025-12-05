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
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../../MainAbility/common/Common';
import {MessageManager,Callback} from '../../../MainAbility/common/MessageManager';


export default function getListItemRect6() {
  describe('getListItemRect6', () => {
    beforeEach(async (done: Function) => {
      console.info("getListItemRect6 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/getItemRect/list/getListItemRect6',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get getListItemRect6 state pages: " + JSON.stringify(pages));
        if (!("getListItemRect6" == pages.name)) {
          console.info("get getListItemRect6 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push getListItemRect6 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getListItemRect6 page error: " + err);
        expect().assertFail();
      }
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back({url: 'MainAbility/pages/index/index'})
      console.info("BadgeFocus after each called");
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getListItemRect6_0100
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 3,Delete listItem3
    * @tc.level : Level 2,Delete listItem3
    */
    it('getListItemRect6_0100', Level.LEVEL0, async (done: Function) => {

      console.info('[getListItemRect6_0100] START');
      globalThis.value.message.notify({name:'index', value:3});
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let listItem_4 = CommonFunc.getComponentRect('getListItemRect6_listItem_4');


      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      console.log('delete')
      await driver.swipe(
        Math.round((listItem_4.right-listItem_4.left)*0.9+listItem_4.left + globalThis.winLeft),
        Math.round((listItem_4.bottom-listItem_4.top)*0.5+listItem_4.top + globalThis.winTop),
        Math.round((listItem_4.right-listItem_4.left)*0.1+listItem_4.left + globalThis.winLeft),
        Math.round((listItem_4.bottom-listItem_4.top)*0.5+listItem_4.top + globalThis.winTop)
      )
      await CommonFunc.sleep(1000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getListItemRect6_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getListItemRect6_text1');
      let text2 = CommonFunc.getComponentRect('getListItemRect6_text2');
      let text3 = CommonFunc.getComponentRect('getListItemRect6_text3');
      let text4 = CommonFunc.getComponentRect('getListItemRect6_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getListItemRect6_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getListItemRect6_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getListItemRect6_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getListItemRect6_text4')
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

  })
}
