
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
import events_emitter from '@ohos.events.emitter';
import Utils from '../Utils';

export default function layoutConstraintsJsunit() {
  describe('layoutConstraintsTest', () => {
    beforeEach(async (done: Function) => {
      console.info("layoutConstraints beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/layoutConstraints/layoutConstraints',
      }
      try {
        router.clear();
        let pages: router.RouterState = router.getState();
        console.info("get layoutConstraints state pages:" + JSON.stringify(pages));
        if (!("layoutConstraints" == pages.name)) {
          console.info("get layoutConstraints state pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await Utils.sleep(2000);
          console.info("push layoutConstraints page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push layoutConstraints page error:" + err);
      }
      done()
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("layoutConstraints after each called");
    });

    it('testLayoutConstraints001', Level.LEVEL0, async (done: Function) => {
      console.info('[testLayoutConstraints001] START');
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('aspectRatio');
      console.info("[testLayoutConstraints001] component strJson:" + strJson);
      console.info("[testLayoutConstraints001] aspectRatio: " + JSON.parse(strJson).$attrs.aspectRatio);
      expect(JSON.parse(strJson).$attrs.aspectRatio).assertEqual(1.5);
      done()
    });

    it('testLayoutConstraints002', Level.LEVEL0, async (done: Function) => {
      console.info('[testLayoutConstraints002] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "aspectRatio": 2
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 20,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testLayoutConstraints002] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testLayoutConstraints002] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('aspectRatio');
      console.info("[testLayoutConstraints002] component strJson:" + strJson);
      console.info("[testLayoutConstraints002] aspectRatio: " + JSON.parse(strJson).$attrs.aspectRatio);
      expect(JSON.parse(strJson).$attrs.aspectRatio).assertEqual(2);
      done()
    });

    it('testLayoutConstraints003', Level.LEVEL0, async (done: Function) => {
      console.info('[testLayoutConstraints003] START');
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('displayPriority');
      console.info("[testLayoutConstraints003] component strJson:" + strJson);
      console.info("[testLayoutConstraints003] displayPriority: " + JSON.parse(strJson).$attrs.displayPriority);
      expect(JSON.parse(strJson).$attrs.displayPriority).assertEqual(1);
      done()
    });

    it('testLayoutConstraints004', Level.LEVEL0, async (done: Function) => {
      console.info('[testLayoutConstraints004] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "displayPriority": 5
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 21,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testLayoutConstraints004] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testLayoutConstraints004] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson = getInspectorByKey('displayPriority');
      console.info("[testLayoutConstraints004] component strJson:" + strJson);
      console.info("[testLayoutConstraints004] displayPriority: " + JSON.parse(strJson).$attrs.displayPriority);
      expect(JSON.parse(strJson).$attrs.displayPriority).assertEqual(1);
      done();
    });

    it('testLayoutConstraints005', Level.LEVEL0, async (done: Function) => {
      console.info('[testLayoutConstraints005] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "aspectRatio": -4
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 1112,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testLayoutConstraints005] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testLayoutConstraints005] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('aspectRatio');
      console.info("[testLayoutConstraints005] component strJson:" + strJson);
      console.info("[testLayoutConstraints005] aspectRatio: " + JSON.parse(strJson).$attrs.aspectRatio);
      expect(JSON.parse(strJson).$attrs.aspectRatio).assertEqual("");
      done()
    });

    it('testLayoutConstraints006', Level.LEVEL0, async (done: Function) => {
      console.info('[testLayoutConstraints006] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "aspectRatio": 9997077
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 1112,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testLayoutConstraints006] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testLayoutConstraints006] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('aspectRatio');
      console.info("[testLayoutConstraints006] component strJson:" + strJson);
      console.info("[testLayoutConstraints006] aspectRatio: " + JSON.parse(strJson).$attrs.aspectRatio);
      expect(JSON.parse(strJson).$attrs.aspectRatio).assertEqual(9997077);
      done()
    });
  });
}