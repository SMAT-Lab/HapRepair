/*
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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import events_emitter from '@ohos.events.emitter'

export default function NavDestinationUIObserver() {
  describe('NavDestinationUIObserver', () => {
    beforeEach(async (done: Function) => {
      console.info("NavDestinationUIObserver beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/NavDestinationUIObserver',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavDestinationUIObserver state pages:" + JSON.stringify(pages));
        if (!("NavDestinationUIObserver" == pages.name)) {
          console.info("get NavDestinationUIObserver pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavDestinationUIObserver page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavDestinationUIObserver page error:" + err);
      }
      console.info("NavDestinationUIObserver beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavDestinationUIObserver after each called");
    });


    /**
     * @tc.number    NavDestinationUIObserver_0100
     * @tc.name      NavDestinationUIObserver_0100
     * @tc.desc      Test UIObserver on_navDestinationSwitch on_navDestinationUpdate
     * @tc.level     Level 0
     */
    it('NavDestinationUIObserver_0100', Level.LEVEL0, async (done: Function) => {
      console.info('NavDestinationUIObserver_0100 START');
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('onBtn'));
      await btn1.click();
      console.info('[NavDestinationUIObserver_0100] btn1 click');
      await driver.delayMs(1000);

      let btn2: Component = await driver.findComponent(ON.id('pushId'));
      await btn2.click();
      console.info('[NavDestinationUIObserver_0100] btn2 click');
      await driver.delayMs(2000);

      let showText: string = AppStorage.get('showText') as string
      console.log('NavDestinationUIObserver_0100 on showText value : ' + showText)
      expect(showText).assertEqual('on');
      AppStorage.delete('showText');

      let showUpdateText: string = AppStorage.get('showUpdateText') as string
      console.log('NavDestinationUIObserver_0100 on showText value : ' + showUpdateText)
      expect(showUpdateText).assertEqual('on');
      AppStorage.delete('showUpdateText');

      let backBtn: Component = await driver.findComponent(ON.id('backId'));
      await backBtn.click();
      console.info('[NavDestinationUIObserver_0100] backBtn click');
      await driver.delayMs(1000);

      console.log('[NavDestinationUIObserver_0100] END');
      done();
    });

    /**
     * @tc.number    NavDestinationUIObserver_0200
     * @tc.name      NavDestinationUIObserver_0200
     * @tc.desc      Test UIObserver off_navDestinationSwitch off_navDestinationUpdate
     * @tc.level     Level 0
     */
    it('NavDestinationUIObserver_0200', Level.LEVEL0, async (done: Function) => {
      console.info('NavDestinationUIObserver_0200 START');
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('offBtn'));
      await btn1.click();
      console.info('[NavDestinationUIObserver_0200] btn1 click');
      await driver.delayMs(1000);

      let btn2: Component = await driver.findComponent(ON.id('pushId'));
      await btn2.click();
      console.info('[NavDestinationUIObserver_0200] btn2 click');
      await driver.delayMs(2000);

      let showText: string = AppStorage.get('showText') as string
      console.log('NavDestinationUIObserver_0200 on showText value : ' + showText)
      expect(showText).assertEqual('off');
      AppStorage.delete('showText');

      let showUpdateText: string = AppStorage.get('showUpdateText') as string
      console.log('NavDestinationUIObserver_0200 on showText value : ' + showUpdateText)
      expect(showUpdateText).assertEqual('off');
      AppStorage.delete('showUpdateText');

      console.log('[NavDestinationUIObserver_0200] END');
      done();
    });

  })
}