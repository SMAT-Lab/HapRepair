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
import CommonFunc from '../../MainAbility/common/Common';
import { MessageManager, Callback } from '../../MainAbility/common/MessageManager';


export default function onPlaceChildren1() {
  describe('onPlaceChildren1', () => {
    beforeEach(async (done: Function) => {
      console.info("onPlaceChildren1 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/PlaceChildren/onPlaceChildren1',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get onPlaceChildren1 state pages: " + JSON.stringify(pages));
        if (!("onPlaceChildren1" == pages.name)) {
          console.info("get onPlaceChildren1 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push onPlaceChildren1 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push onPlaceChildren1 page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_onPlaceChildren1_001
    * @tc.name  : onMeasureSize
    * @tc.desc  : Get minHeight,maxHeight,minWidth,maxWidth from onPlaceChildren
    * @tc.level : Level 2
    */
    it('onPlaceChildren1_001', Level.LEVEL0, async (done: Function) => {
      console.info('[onPlaceChildren1_001] START');
      await CommonFunc.sleep(500);
      let MaxHeightText1 : string = JSON.parse(getInspectorByKey('constraint_maxHeight1')).$attrs.content;
      await CommonFunc.sleep(500);
      console.log('maxHeight111 is ' + MaxHeightText1);
      expect(Math.abs(Number(MaxHeightText1) - 600) <= 1).assertTrue();
      await CommonFunc.sleep(500);
      let MinHeightText1 : string = JSON.parse(getInspectorByKey('constraint_minHeight1')).$attrs.content;
      await CommonFunc.sleep(500);
      console.log('minHeight111 is ' + MinHeightText1);
      expect(Math.abs(Number(MinHeightText1) - 0) <= 1).assertTrue();
      await CommonFunc.sleep(500);
      let MaxWidthText1 : string = JSON.parse(getInspectorByKey('constraint_maxWidth1')).$attrs.content;
      await CommonFunc.sleep(500);
      console.log('maxWidth111 is' + MaxWidthText1);
      expect(Math.abs(Number(MaxWidthText1) - 500) <= 1).assertTrue();
      await CommonFunc.sleep(500);
      let MinWidthText1 : string = JSON.parse(getInspectorByKey('constraint_minWidth1')).$attrs.content;
      await CommonFunc.sleep(500);
      console.log('minWidth111 is' + MinWidthText1);
      expect(Math.abs(Number(MinWidthText1) - 0) <= 1).assertTrue();
      await CommonFunc.sleep(500);
      console.info('[onPlaceChildren1_001] END');
      done();
    });
  })
}
