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
import CommonFunc from '../../MainAbility/common/Common';
import { MessageManager, Callback } from '../../MainAbility/common/MessageManager';


export default function onMeasureSize3() {
  describe('onMeasureSize3', () => {
    beforeEach(async (done: Function) => {
      console.info("onMeasureSize3 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/MeasureSize/onMeasureSize3',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get onMeasureSize3 state pages: " + JSON.stringify(pages));
        if (!("onMeasureSize3" == pages.name)) {
          console.info("get onMeasureSize3 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push onMeasureSize3 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push onMeasureSize3 page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_onMeasureSize3_001
    * @tc.name  : onMeasureSize
    * @tc.desc  : Set margin and padding when calling onMeasureSize and onPlaceChildren interfaces normally
    * @tc.level : Level 2
    */
    it('onMeasureSize3_001', Level.LEVEL0, async (done: Function) => {
      console.info('[onMeasureSize3_001] START');
      await CommonFunc.sleep(1000);

      console.log('get Initial value')
      let text_01 = CommonFunc.getComponentRect('onMeasureSize3_text_01');
      let text_02 = CommonFunc.getComponentRect('onMeasureSize3_text_02');
      let text_03 = CommonFunc.getComponentRect('onMeasureSize3_text_03');
      let Column_1 = CommonFunc.getComponentRect('onMeasureSize3_Column_1');

      console.log('assert position')
      //expect(Math.round(text_01.left - Column_1.left)).assertEqual(Math.round(vp2px(230)));
      expect(Math.abs(Math.round(text_01.left - Column_1.left)-Math.round(vp2px(230)))<=1).assertTrue()
      //expect(Math.round(text_01.top - Column_1.top)).assertEqual(Math.round(vp2px(230)));
      expect(Math.abs(Math.round(text_01.top - Column_1.top)-Math.round(vp2px(230)))<=1).assertTrue()
      console.log('text_01.left' + text_01.left)
      console.log('text_01.top' + text_01.top)
      //expect(Math.round(text_02.left - Column_1.left)).assertEqual(Math.round(vp2px(150)));
      expect(Math.abs(Math.round(text_02.left - Column_1.left)-Math.round(vp2px(150)))<=1).assertTrue()
      //expect(Math.round(text_02.top - Column_1.top)).assertEqual(Math.round(vp2px(150)));
      expect(Math.abs(Math.round(text_02.top - Column_1.top)-Math.round(vp2px(150)))<=1).assertTrue()
      console.log('text_02.left' + text_01.left)
      console.log('text_02.top' + text_01.top)
      console.log('Column_1.left' + Column_1.left)
      console.log('Column_1.top' + Column_1.top)
      console.log('assert size')
      //expect(Math.round(text_01.right - text_01.left)).assertEqual(Math.round(vp2px(100)));
      expect(Math.abs(Math.round(text_01.right - text_01.left)-Math.round(vp2px(100)))<=1).assertTrue()
      console.log('text_01.right - text_01.left' + (text_01.right - text_01.left))
      //expect(Math.round(text_01.bottom - text_01.top)).assertEqual(Math.round(vp2px(100)));
      expect(Math.abs(Math.round(text_01.bottom - text_01.top)-Math.round(vp2px(100)))<=1).assertTrue()
      console.log('text_01.bottom - text_01.top' + (text_01.bottom - text_01.top))
      //expect(Math.round(text_02.right - text_02.left)).assertEqual(Math.round(vp2px(150)));
      expect(Math.abs(Math.round(text_02.right - text_02.left)-Math.round(vp2px(150)))<=1).assertTrue()
      console.log('text_02.right - text_02.left' + (text_02.right - text_02.left))
      //expect(Math.round(text_02.bottom - text_02.top)).assertEqual(Math.round(vp2px(150)));
      expect(Math.abs(Math.round(text_02.bottom - text_02.top)-Math.round(vp2px(150)))<=1).assertTrue()
      console.log('text_02.bottom - text_02.top' + (text_02.bottom - text_02.top))
      done();
    });
  })
}
