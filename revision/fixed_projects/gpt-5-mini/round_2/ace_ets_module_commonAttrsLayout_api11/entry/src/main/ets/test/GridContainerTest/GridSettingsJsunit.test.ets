
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

export default function gridSettingsJsunit() {
  describe('gridSettingsTest', () => {
    beforeEach(async (done:Function) => {
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/GridContainer/gridSettings',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get gridSettings state pages:" + JSON.stringify(pages));
        if (!("textStyle" == pages.name)) {
          console.info("get gridSettings state pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await Utils.sleep(2000);
          console.info("push gridSettings page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push gridSettings page error:" + err);
      }
      done()
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("gridSettings after each called");
    });

    it('testGridSettings000', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings000] START');
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('gridContainer');
      console.info("[testGridSettings000] component strJson:" + strJson);
      expect(true).assertEqual(true);
      done()
    });

    it('testGridSettings001', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings001] START');
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridRow');
      console.info("[testGridSettings001] component strJson:" + strJson);

      console.info("[testGridSettings001] gridRow useSizeType: " + JSON.stringify(JSON.parse(strJson).$attrs.useSizeType));
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.span).assertEqual(2);
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.offset).assertEqual(1);
      done()
    });

    it('testGridSettings002', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings002] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "span": 1,
            "offset": 2
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 9,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings002] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings002] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson = getInspectorByKey('gridRow');
      console.info("[testGridSettings002] component strJson:" + strJson);

      console.info("[testGridSettings002] gridRow useSizeType: " + JSON.stringify(JSON.parse(strJson)
        .$attrs
        .useSizeType));
      expect(JSON.parse(strJson)
        .$attrs
        .useSizeType
        .sm
        .span).assertEqual(1);
      expect(JSON.parse(strJson)
        .$attrs
        .useSizeType
        .sm
        .offset).assertEqual(2);
      done()
    });

    it('testGridSettings003', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings003] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "span": 0x80000000,
            "offset": 0x80000000
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 9,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings003] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings003] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridRow');
      console.info("[testGridSettings003] component strJson:" + strJson);

      console.info("[testGridSettings003] gridRow useSizeType: " + JSON.stringify(JSON.parse(strJson).$attrs.useSizeType));
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.span).assertEqual(1);
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.offset).assertEqual(-2147483648);
      done()
    });

    it('testGridSettings004', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings004] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "span": 0x7FFFFFFF,
            "offset": 0x7FFFFFFF
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 9,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings004] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings004] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridRow');
      console.info("[testGridSettings004] component strJson:" + strJson);

      console.info("[testGridSettings004] gridRow useSizeType: " + JSON.stringify(JSON.parse(strJson).$attrs.useSizeType));
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.span).assertEqual(0x7FFFFFFF);
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.offset).assertEqual(0x7FFFFFFF);
      done()
    });

    it('testGridSettings005', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings005] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "span": 'error',
            "offset": 'error'
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 9,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings005] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings005] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridRow');
      console.info("[testGridSettings005] component strJson:" + strJson);

      console.info("[testGridSettings005] gridRow useSizeType: " + JSON.stringify(JSON.parse(strJson).$attrs.useSizeType));
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.span).assertEqual(0);
      expect(JSON.parse(strJson).$attrs.useSizeType.sm.offset).assertEqual(0);
      done()
    });

    it('testGridSettings006', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings006] START');
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('gridOffset');
      console.info("[testGridSettings006] component strJson:" + strJson);

      console.info("[testGridSettings006] gridOffset: " + JSON.stringify(JSON.parse(strJson).$attrs.gridOffset));
      expect(JSON.parse(strJson).$attrs.gridOffset).assertEqual(0);
      done()
    });

    it('testGridSettings007', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings007] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridOffset": 1
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 10,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings007] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings007] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridOffset');
      console.info("[testGridSettings007] component strJson:" + strJson);

      console.info("[testGridSettings007] gridOffset: " + JSON.stringify(JSON.parse(strJson).$attrs.gridOffset));
      expect(JSON.parse(strJson).$attrs.gridOffset).assertEqual(1);
      done()
    });

    it('testGridSettings008', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings008] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridOffset": 0x80000000
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 10,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings008] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings008] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridOffset');
      console.info("[testGridSettings008] component strJson:" + strJson);

      console.info("[testGridSettings008] gridOffset: " + JSON.stringify(JSON.parse(strJson).$attrs.gridOffset));
      expect(JSON.parse(strJson).$attrs.gridOffset).assertEqual(-2147483648);
      done()
    });

    it('testGridSettings009', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings009] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridOffset": 0x7FFFFFFF
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 10,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings009] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings009] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridOffset');
      console.info("[testGridSettings009] component strJson:" + strJson);

      console.info("[testGridSettings009] gridOffset: " + JSON.stringify(JSON.parse(strJson).$attrs.gridOffset));
      expect(JSON.parse(strJson).$attrs.gridOffset).assertEqual(0x7FFFFFFF);
      done()
    });

    it('testGridSettings010', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings010] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridOffset": 'error'
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 10,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings010] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings010] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridOffset');
      console.info("[testGridSettings010] component strJson:" + strJson);
      console.info("[testGridSettings010] gridOffset: " + JSON.stringify(JSON.parse(strJson).$attrs.gridOffset));
      expect(JSON.parse(strJson).$attrs.gridOffset).assertEqual(0);
      done()
    });

    it('testGridSettings011', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings011] START');
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('gridSpan');
      console.info("[testGridSettings011] component strJson:" + strJson);
      console.info("[testGridSettings011] gridSpan: " + JSON.stringify(JSON.parse(strJson).$attrs.gridSpan));
      expect(JSON.parse(strJson).$attrs.gridSpan).assertEqual(1);
      done()
    });

    it('testGridSettings012', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings012] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridSpan": 2
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 11,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings012] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings012] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridSpan');
      console.info("[testGridSettings012] component strJson:" + strJson);
      console.info("[testGridSettings012] gridSpan: " + JSON.stringify(JSON.parse(strJson).$attrs.gridSpan));
      expect(JSON.parse(strJson).$attrs.gridSpan).assertEqual(2);
      done()
    });

    it('testGridSettings013', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings013] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridSpan": 0x80000000
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 11,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings013] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings013] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridSpan');
      console.info("[testGridSettings013] component strJson:" + strJson);
      console.info("[testGridSettings013] gridSpan: " + JSON.stringify(JSON.parse(strJson).$attrs.gridSpan));
      expect(JSON.parse(strJson).$attrs.gridSpan).assertEqual(2);
      done()
    });

    it('testGridSettings014', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings014] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridSpan": 0x7FFFFFFF
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 11,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings014] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings014] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridSpan');
      console.info("[testGridSettings014] component strJson:" + strJson);
      console.info("[testGridSettings014] gridSpan: " + JSON.stringify(JSON.parse(strJson).$attrs.gridSpan));
      expect(JSON.parse(strJson).$attrs.gridSpan).assertEqual(0x7FFFFFFF);
      done()
    });

    it('testGridSettings015', Level.LEVEL0, async (done:Function) => {
      console.info('[testGridSettings015] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "gridSpan": 'error'
          }
        }
        let indexEvent: events_emitter.InnerEvent = {
          eventId: 11,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGridSettings015] start to publish emit");
        events_emitter.emit(indexEvent, eventData);
      } catch (err) {
        console.log("[testGridSettings015] change component data error: " + err.message);
      }
      await Utils.sleep(4000);
      let strJson: string = getInspectorByKey('gridSpan');
      console.info("[testGridSettings015] component strJson:" + strJson);
      console.info("[testGridSettings015] gridSpan: " + JSON.stringify(JSON.parse(strJson).$attrs.gridSpan));
      expect(JSON.parse(strJson).$attrs.gridSpan).assertEqual(1);
      done()
    });
  });
}