/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import Utils from '../../MainAbility/common/Utils'
import events_emitter from '@ohos.events.emitter';

export default function routerStandardJsunit() {
  describe('routerStandardTest', () => {
    beforeEach(async (done: Function) => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ohosRouter/router',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get router state success " + JSON.stringify(pages));
        if (!("router" == pages.name)) {
          console.info("get router state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await Utils.sleep(2000);
          console.info("push router page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push router page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("routerStandard after each called");
    });

    /**
     * @tc.number    testrouterStandard0011
     * @tc.name      testrouterStandard0011
     * @tc.desc      testrouterStandard0011
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('testrouterStandard0011', Level.LEVEL2, async (done: Function) => {
      console.info('routerStandard testrouterStandard0011 START');
      await Utils.sleep(2000);
      try {
        let standardEvent: events_emitter.InnerEvent = {
          eventId: 101,
          priority: events_emitter.EventPriority.LOW
        }
        let callback1 = (eventData: events_emitter.EventData) => {
          console.info("testrouterStandard_0011 get event data is: " + JSON.stringify(eventData));
          expect(eventData.data?.ArrayData).assertEqual("45");

        }
        events_emitter.on(standardEvent, callback1);
        console.info("testrouterStandard_0011 click result1 is: " + JSON.stringify(sendEventByKey('StandardText', 10, "")));
        await Utils.sleep(2000);
        console.info("testrouterStandard_0011 click result2 is: " + JSON.stringify(sendEventByKey('DataText', 10, "")));
        await Utils.sleep(2000);
      } catch (err) {
        console.info("testrouterStandard_0011 on events_emitter err : " + JSON.stringify(err));
      }
      console.info('testrouterStandard0011 END');
      done();
    });

    /**
     * @tc.number    testrouterSingle0001
     * @tc.name      testrouterSingle0001
     * @tc.desc      testrouterSingle0001
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('testrouterSingle0001', Level.LEVEL2, async (done: Function) => {
      console.info('routerSingle testrouterSingle0001 START');
      await Utils.sleep(2000);
      try {
        let singleEvent: events_emitter.InnerEvent = {
          eventId: 102,
          priority: events_emitter.EventPriority.LOW
        }
        let callback2 = (eventData: events_emitter.EventData) => {
          console.info("testrouterSingle_0001 get event data is: " + JSON.stringify(eventData));
          expect(eventData.data?.ArrayData).assertEqual("46");
        }
        events_emitter.on(singleEvent, callback2);
        console.info("testrouterSingle_0001 click result1 is: " + JSON.stringify(sendEventByKey('SingleText', 10, "")));
        await Utils.sleep(2000);
        console.info("testrouterSingle_0001 click result2 is: " + JSON.stringify(sendEventByKey('SingleDataText', 10, "")));
        await Utils.sleep(2000);
      } catch (err) {
        console.info("testrouterSingle_0001 on events_emitter err : " + JSON.stringify(err));
      }
      console.info('testrouterSingle_0001 END');
      done();
    });
  })
}
