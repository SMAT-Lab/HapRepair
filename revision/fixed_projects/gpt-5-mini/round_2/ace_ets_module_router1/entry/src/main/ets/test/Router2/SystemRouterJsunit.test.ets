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
import { describe, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import Utils from '../../MainAbility/common/Utils';
import events_emitter from '@ohos.events.emitter';

export default function systemRouterJsunit() {
  describe('systemRouterTest', () => {
    /**
     * @tc.number systemRouterTest001
     * @tc.name systemRouterTest001
     * @tc.desc systemRouterTest001
     */
    it('systemRouterTest001', Level.LEVEL2, async (done: Function) => {
      console.info('[systemRouterTest001] START');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/systemRouterA',
      }
      try {
        let result = await router.pushUrl(options);
        await Utils.sleep(1000);
        console.info("[systemRouterTest001] push page success " + JSON.stringify(result));
        let pages = router.getState();
        expect(pages.name).assertEqual('systemRouterA');
        expect(pages.path).assertEqual('MainAbility/pages/Router2/');
      } catch (err) {
        console.error("[systemRouterTest001] push page error: " + err);
      }
      await Utils.sleep(2000);
      done();
    });

    /**
     * @tc.number systemRouterTest002
     * @tc.name systemRouterTest002
     * @tc.desc systemRouterTest002
     */
    it('systemRouterTest002', Level.LEVEL2, async (done: Function) => {
      console.info('[systemRouterTest002] START');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/systemRouterA'
      }
      try {
        let result = await router.pushUrl(options);
        console.info("[systemRouterTest002] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[systemRouterTest002] push page error " + JSON.stringify(err));
      }
      await Utils.sleep(1000);
      router.clear();
      await Utils.sleep(1500);
      let callback = (paramsEvent: events_emitter.EventData) => {
        console.info("[systemRouterTest002] get state result is: " + JSON.stringify(paramsEvent));
        expect(paramsEvent.data?.params).assertEqual('B Page');
      }
      let paramsEvent: events_emitter.InnerEvent = {
        eventId: 211,
        priority: events_emitter.EventPriority.LOW
      }
      try {
        events_emitter.on(paramsEvent, callback);
      } catch (err) {
        console.info("[systemRouterTest002] on events_emitter err : " + JSON.stringify(err));
      }
      console.info("[systemRouterTest002] click push button result is: "
        + JSON.stringify(sendEventByKey('pushButtonA', 10, "")));
      await Utils.sleep(1000);
      router.back();
      await Utils.sleep(1000);
      done();
    });

    /**
     * @tc.number systemRouterTest003
     * @tc.name systemRouterTest003
     * @tc.desc systemRouterTest003
     */
    it('systemRouterTest003', Level.LEVEL2, async (done: Function) => {
      console.info('[systemRouterTest003] START');
      await Utils.sleep(1000);
      router.clear();
      await Utils.sleep(2000);
      let size = router.getLength();
      console.info("[systemRouterTest003] size" + size);
      expect(size).assertEqual("1");
      await Utils.sleep(1000);
      done();
    });

    /**
     * @tc.number systemRouterTest004
     * @tc.name systemRouterTest004
     * @tc.desc systemRouterTest004
     */
    it('systemRouterTest004', Level.LEVEL2, async (done: Function) => {
      console.info('[systemRouterTest004] START');
      router.clear();
      await Utils.sleep(1000);
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/systemRouterA'
      }
      try {
        let result = await router.pushUrl(options);
        console.info("[systemRouterTest004] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[systemRouterTest004] push page error " + JSON.stringify(err));
      }
      await Utils.sleep(1000);
      let optionsB: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/systemRouterB'
      }
      try {
        let result = router.replaceUrl(optionsB);
        console.info("[systemRouterTest004] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[systemRouterTest004] push page error " + JSON.stringify(err));
      }
      await Utils.sleep(1500);
      let size = router.getLength();
      console.info("[systemRouterTest004] size" + size);
      expect(size).assertEqual("2");
      done();
    });

    /**
     * @tc.number systemRouterTest005
     * @tc.name systemRouterTest005
     * @tc.desc systemRouterTest005
     */
    it('systemRouterTest005', Level.LEVEL2, async (done: Function) => {
      console.info('[systemRouterTest005] START');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/systemRouterB'
      }
      try {
        let result = await router.pushUrl(options);
        console.info("[systemRouterTest005] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[systemRouterTest005] push page error " + JSON.stringify(err));
      }
      await Utils.sleep(1000);
      router.clear();
      await Utils.sleep(1500);
      let callback = (paramsEvent: events_emitter.EventData) => {
        console.info("[systemRouterTest005] get state result is: " + JSON.stringify(paramsEvent));
        expect(paramsEvent.data?.params).assertEqual('B Page');

      }
      let paramsEvent: events_emitter.InnerEvent = {
        eventId: 211,
        priority: events_emitter.EventPriority.LOW
      }
      try {
        events_emitter.on(paramsEvent, callback);
      } catch (err) {
        console.info("[systemRouterTest005] on events_emitter err : " + JSON.stringify(err));
      }
      console.info("[systemRouterTest005] click replace button result is: "
        + JSON.stringify(sendEventByKey('ReplaceButtonA', 10, "")));
      await Utils.sleep(1500);
      done();
    });
  });
}

