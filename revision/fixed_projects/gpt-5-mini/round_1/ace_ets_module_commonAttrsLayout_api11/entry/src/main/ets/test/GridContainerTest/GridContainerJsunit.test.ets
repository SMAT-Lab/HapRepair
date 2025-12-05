
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

export default function girdContainerJsunit() {
  describe('girdContainerTest', () => {
    beforeEach(async (done:Function) => {
      console.info("girdContainer beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/GridContainer/gridContainer'
      }
      try {
        router.clear();
        let pages: router.RouterState = router.getState();
        console.info("get girdContainer state pages:" + JSON.stringify(pages));
        if (!("gridContainer" == pages.name)) {
          console.info("get girdContainer state pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await Utils.sleep(2000);
          console.info("push girdContainer page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push girdContainer page error:" + err);
      }
      done()
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("girdContainer after each called");
    });

    it('testGirdContainer01', Level.LEVEL0, async (done:Function) => {
      console.info('[testGirdContainer01] START');
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('gridContainer');
      console.info("[testGirdContainer01] obj is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$type).assertEqual('GridContainer');
      expect(JSON.parse(strJson).$attrs.constructor.columns).assertEqual('12');
      expect(JSON.parse(strJson).$attrs.constructor.sizeType).assertEqual('SizeType.XS');
      expect(JSON.parse(strJson).$attrs.constructor.gutter).assertEqual('10.00vp');
      expect(JSON.parse(strJson).$attrs.constructor.margin).assertEqual('20.00vp');
      console.info('testGirdContainer01 END');
      done()
    });


    it('testGirdContainer02', Level.LEVEL0, async (done:Function) => {
      console.info('[testEllipse02] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "sizeType": 'SM'
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 109,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testGirdContainer02] start to publish emit");
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testGirdContainer02] change component data error: " + err.message);
      }
      await Utils.sleep(1000);
      let strJson: string = getInspectorByKey('gridContainer');
      console.info("[testGirdContainer02] obj is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.constructor.sizeType).assertEqual('SizeType.SM');
      console.info('testGirdContainer02 END');
      done()
    });
  })
}
