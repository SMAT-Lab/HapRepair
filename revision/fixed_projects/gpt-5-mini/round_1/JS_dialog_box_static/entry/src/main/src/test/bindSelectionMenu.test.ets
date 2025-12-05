/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { beforeAll, beforeEach, describe, it, TestType, Size, Level, afterEach } from '../../../hypium/index';
import router from '@ohos.router';
import Utils from './Util.test';
import abilityDelegatorRegistry from '@ohos.app.ability.abilityDelegatorRegistry';
import inspector from '@ohos.arkui.inspector';
import UIAbility from '@ohos.app.ability.UIAbility';
import { BusinessError } from '@ohos.base';
import common from '@ohos.app.ability.common';
import { Driver, ON } from '@ohos.UiTest';

let testAbilityContext:common.UIAbilityContext;

export default function webViewSrollableJsunit() {
  describe('bindSelectionMenuTest',  (): void => {
    beforeAll(async (): Promise<void> => {
      await Utils.msSleep(2000)
      console.log("App start")
    });

    beforeEach(async () => {
      await Utils.msSleep(3000);
      console.info("web beforeEach start");
    })

    /*
    * @tc.number     : SUB_ACE_BASIC_ETS_BINDSELECTIONMENUTYPE_API_001
    * @tc.name       : bindSelectionMenuTypeStatic01
    * @tc.desc       : test bindSelectionMenuType01 interface when choose callback mode.
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('bindSelectionMenuTypeStatic01',TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3, async (done:()=>void):Promise<void> => {
      await Utils.msSleep(1000)
      Utils.emitEvent("bindSelectionMenuType01", 10)
      await Utils.msSleep(1000)
      Utils.registerEvent("bindSelectionMenuType01",true,8301,done);
      await Utils.msSleep(1000)

      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('bindSelectionMenuType'));
      button?.longClick();
      await Utils.msSleep(2000);
      let buttonOne = await driver.findComponent(ON.text('复制'));
      buttonOne?.click()
      await Utils.msSleep(2000);
      inspector.sendEventByKey('webcomponentSelectionMenu',10,'');
      done();
    })

    /*
    * @tc.number     : SUB_ACE_BASIC_ETS_BINDSELECTIONMENUTYPE_API_002
    * @tc.name       : bindSelectionMenuTypeStatic02
    * @tc.desc       : test bindSelectionMenuType02 interface when choose callback mode.
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('bindSelectionMenuTypeStatic02',TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3, async (done:()=>void):Promise<void> => {
      await Utils.msSleep(1000)
      Utils.emitEvent("bindSelectionMenuType02", 10)
      await Utils.msSleep(1000)
      Utils.registerEvent("bindSelectionMenuType02",1,8302,done);
      await Utils.msSleep(1000)
      inspector.sendEventByKey('webcomponentSelectionMenu',10,'');
      await Utils.msSleep(1000)
      done();
    })

    it('SwitchNextPageStatic', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        await Utils.msSleep(1000);
        Utils.emitEvent('SwitchNextPageStatic', 10)
        await Utils.msSleep(1000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(1000);
        let button = await driver.findComponent(ON.id('webcomponentSelectionMenu'));
        await Utils.msSleep(1000);
        button?.click();
        await Utils.msSleep(4000)
        done()
      })
  })
}
