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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium";
import router from '@ohos.router';
import CommonFunc from '../../MainAbility/common/Common';

interface GeneratedObjectLiteralInterface_1 {
  url: string;
}

export default function marqueeOptionsTest1() {
  describe('marqueeOptionsTest1', () => {
    beforeEach(async (done: Function) => {
      console.info("marqueeOptionsTest1 beforeEach start");
      let options: GeneratedObjectLiteralInterface_1 = {
        url: "MainAbility/pages/Text/marqueeOptions1",
      };
      try {
        router.clear();
        let pages = router.getState();
        console.info("get marqueeOptionsTest1 state pages:" + JSON.stringify(pages));
        if (!("marqueeOptionsTest1" == pages.name)) {
          console.info("get marqueeOptionsTest1 pages.name:" + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push marqueeOptionsTest1 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push marqueeOptionsTest1 page error:" + err);
      }
      console.info("marqueeOptionsTest1 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("marqueeOptionsTest1 after each called");
    });
	
    /**
     * @tc.number marqueeOptionsTest1
     * @tc.name testBack
     * @tc.desc Return specified page by index value
     * @tc.level Level 3
     * @tc.type
     * @tc.size
     */
    it('marqueeOptionsTest1_001', Level.LEVEL3, async (done: Function) => {
      console.info('[marqueeOptionsTest1_001] START');
      await CommonFunc.sleep(1000);
      let marqueeOptions = getInspectorByKey('MarqueeOptions');
      console.info('marqueeOptions-----' + JSON.parse(marqueeOptions).$attrs.marqueeOptions);
      await CommonFunc.sleep(2000);
      let marq: ESObject = JSON.parse(JSON.parse(marqueeOptions).$attrs.marqueeOptions);
      expect(marq.start).assertTrue();
      expect(Math.abs(marq.step - vp2px(6)) <= 1).assertTrue();
      expect(marq.loop == -1).assertTrue();
      expect(marq.delay == 0).assertTrue();
      expect(marq.fadeout).assertFalse();
      expect(marq.startPolicy).assertEqual('MarqueeStartPolicy.DEFAULT');
      expect(marq.direction).assertEqual('MarqueeDirection.DEFAULT');
      console.info('[marqueeOptionsTest1_001] END');
      done();
    });
    
    it('onMarqueeStateChangeTest_001', Level.LEVEL3, async (done: Function) => {
      console.info('[onMarqueeStateChangeTest_001] START');
      await CommonFunc.sleep(1000);
      let MarqueeState = getInspectorByKey('MarqueeState');
      await CommonFunc.sleep(2000);
      console.info('MarqueeState-----' + JSON.parse(MarqueeState).$attrs.content);
      await CommonFunc.sleep(1000);
      expect(JSON.parse(MarqueeState).$attrs.content).assertEqual('START');
      console.info('[onMarqueeStateChangeTest_001] END');
      done();
    });
  })
}