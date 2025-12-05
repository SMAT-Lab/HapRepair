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
import CommonFunc from '../../MainAbility/common/Common';
import events_emitter from '@ohos.events.emitter';

export default function edgeEffectJsunit() {
  describe('edgeEffectTest', () => {
    beforeEach(async (done: Function) => {
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/edgeEffect',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get edgeEffect state success " + JSON.stringify(pages));
        if (!("edgeEffect" == pages.name)) {
          console.info("get edgeEffect state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(1000);
          console.info("push edgeEffect page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push edgeEffect page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("edgeEffect after each called");
    });

    /**
     * @tc.number    edgeEffectTest_0100
     * @tc.name      edgeEffectTest_0100
     * @tc.desc      Testing edgeEffect for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('edgeEffectTest_0100', Level.LEVEL1, async (done: Function) => {
      console.info('edgeEffectTest_0100 START');
      let strJson: string = getInspectorByKey('list_eff');
      console.info("edgeEffectTest_0100 component strJson:" + strJson);
      console.info("edgeEffectTest_0100 component obj is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.edgeEffect).assertEqual("EdgeEffect.None");
      done();
    });

    /**
     * @tc.number    edgeEffectTest_0200
     * @tc.name      edgeEffectTest_0200
     * @tc.desc      Testing edgeEffect for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('edgeEffectTest_0200', Level.LEVEL1, async (done: Function) => {
      console.info('edgeEffectTest_0200 START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "edgeEffect": EdgeEffect.Spring,
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 91,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("edgeEffectTest_0200 start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("edgeEffectTest_0200 change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJsonNew: string = getInspectorByKey('list_eff');
      console.info("edgeEffectTest_0200 component objNew is: " + JSON.stringify(JSON.parse(strJsonNew)));
      expect(JSON.parse(strJsonNew).$attrs.edgeEffect).assertEqual('EdgeEffect.Spring');
      console.info('edgeEffectTest_0200 END');
      done();
    });
  })
}