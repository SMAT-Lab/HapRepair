/**
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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Driver, Component, UiWindow, ON } from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';
import events_emitter from '@ohos.events.emitter';

export default function chainAnimationList() {
  describe('chainAnimationList', () => {
    beforeEach(async (done: Function) => {
      console.info("chainAnimationList beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/chainAnimationList",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get chainAnimationList state pages:" + JSON.stringify(pages));
        if (!("chainAnimationList" == pages.name)) {
          console.info("get chainAnimationList pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push chainAnimationList page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push chainAnimationList page error:" + err);
      }
      console.info("chainAnimationList beforeEach end");
      done();
    });

    /**
     * @tc.number    chainAnimationList_001
     * @tc.name      chainAnimationList_0100
     * @tc.desc      ChainAnimationOptions: Set edgeEffect to STRETCH.
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('chainAnimationList_0100', Level.LEVEL0, async (done: Function) => {
      console.info('[chainAnimationList_0100] START');
      await CommonFunc.sleep(1000);
      let driver = await Driver.create()
      let strJson = getInspectorByKey('list_chain');
      console.info("[chainAnimationList_0100] component obj is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[chainAnimationList_0100] chainAnimationOptions: " + JSON.stringify(JSON.parse(strJson).$attrs.chainAnimationOptions));
      console.info("[chainAnimationList_0100] cachedCount: " + JSON.stringify(JSON.parse(strJson).$attrs.cachedCount));
      console.info("[chainAnimationList_0100] scrollBar: " + JSON.stringify(JSON.parse(strJson).$attrs.scrollBar));
      await CommonFunc.sleep(1000);
      expect(JSON.parse(strJson).$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.friction).assertEqual(0.6);
      expect(JSON.parse(strJson).$attrs.cachedCount).assertEqual('2');
      expect(JSON.parse(strJson).$attrs.scrollBar).assertEqual("BarState.Auto");
      await driver.assertComponentExist(ON.text('1'));

      let strJson2 = getInspectorByKey('listItem_sticky');
      console.info("[chainAnimationList_0100] component obj2 is: " + JSON.stringify(JSON.parse(strJson2)));
      console.info("[chainAnimationList_0100] sticky: " + JSON.stringify(JSON.parse(strJson2).$attrs.sticky));
      console.info("[chainAnimationList_0100] editable: " + JSON.stringify(JSON.parse(strJson2).$attrs.editable));
      await CommonFunc.sleep(1000);
      expect(JSON.parse(strJson2).$type).assertEqual('ListItem');
      expect(JSON.parse(strJson2).$attrs.sticky).assertEqual("Sticky.None");
      console.info('[chainAnimationList_0100] END');
      done();
    });
  })
}
