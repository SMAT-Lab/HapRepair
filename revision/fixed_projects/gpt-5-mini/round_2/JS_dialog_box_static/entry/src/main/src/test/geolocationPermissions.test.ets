/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
import { beforeAll, beforeEach, describe, it, TestType, Size, Level } from '../../../hypium/index';
import router from '@ohos.router';
import Utils from './Util.test';
import inspector from '@ohos.arkui.inspector';
import UIAbility from '@ohos.app.ability.UIAbility';
import { BusinessError } from '@ohos.base';
import common from '@ohos.app.ability.common';
import abilityDelegatorRegistry from '@ohos.app.ability.abilityDelegatorRegistry';
import { Driver, ON } from '@ohos.UiTest';

let testAbilityContext: common.UIAbilityContext;

export default function geolocationPermissions() {

  describe('GeolocationPermissionsTestSuite', (): void => {

    beforeAll(async (): Promise<void> => {
      console.log('App start')
      let abilityDelegator:abilityDelegatorRegistry.AbilityDelegator = abilityDelegatorRegistry.getAbilityDelegator()
      abilityDelegator.addAbilityMonitor({
        abilityName: 'EntryAbility',
        moduleName:'entry',
        onAbilityCreate: (abilitys : UIAbility) : void => {
          testAbilityContext = abilitys.context
          console.log('onAbilityCreate end')
        },
      }, (err : BusinessError|null) : void => {
        if (err != null ) {

          console.log('-----' + err.code)
        }

        console.log('BusinessError  end')
      })
      await Utils.msSleep(2000)
      abilityDelegator.executeShellCommand('aa start -a EntryAbility -b com.open.harmony.jsdialogboxtest.static')
        .then(result => {
          console.info('Command executed successfully:', result);
        })
        .catch(err => {
          console.error('Error executing command:', err);
        });
      await Utils.msSleep(2000)
      console.log('beforeAll end,%s', 'Succeed!')
    });

    beforeEach(async () => {
      await Utils.msSleep(2000);
      console.info('web beforeEach start');
    })

    /*
    * @tc.number     : SUB_WEB_DELETEALLGEOLOCATION_STATIC_0100
    * @tc.name       : testDeleteAllGeolocationStatic001
    * @tc.desc       : test deleteAllGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testDeleteAllGeolocationStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testDeleteAllGeolocation001', 10)
        await Utils.msSleep(3000);
        Utils.registerEvent('testDeleteAllGeolocation001', '', 30055, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })

    /*
    * @tc.number     : SUB_WEB_GETSTOREDGEOLOCATION_STATIC_0300
    * @tc.name       : testGetStoredGeolocationStatic003
    * @tc.desc       : test getStoredGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testGetStoredGeolocationStatic003', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testGetStoredGeolocation003', 10)
        await Utils.msSleep(2000);
        Utils.registerEvent('testGetStoredGeolocation003', '', 100011, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })

    /*
    * @tc.number     : SUB_WEB_GETSTOREDGEOLOCATION_STATIC_0400
    * @tc.name       : testGetStoredGeolocationStatic004
    * @tc.desc       : test getStoredGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testGetStoredGeolocationStatic004', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testGetStoredGeolocation004', 10)
        await Utils.msSleep(2000);
        Utils.registerEvent('testGetStoredGeolocation004', '', 100012, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })

    /*
    * @tc.number     : SUB_WEB_GETSTOREDGEOLOCATION_STATIC_0100
    * @tc.name       : testGetStoredGeolocationStatic001
    * @tc.desc       : test getStoredGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testGetStoredGeolocationStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testGetStoredGeolocation001', 10)
        await Utils.msSleep(2000);
        Utils.registerEvent('testGetStoredGeolocation001', 'file:///', 30060, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })

    /*
    * @tc.number     : SUB_WEB_GETSTOREDGEOLOCATION_STATIC_0200
    * @tc.name       : testGetStoredGeolocationStatic002
    * @tc.desc       : test getStoredGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testGetStoredGeolocationStatic002', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testGetStoredGeolocation002', 10)
        await Utils.msSleep(2000);
        Utils.registerEvent('testGetStoredGeolocation002', 'file:///', 30061, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })


    /*
    * @tc.number     : SUB_WEB_GETACCESSIBLEGEOLOCATION_STATIC_0100
    * @tc.name       : testGetAccessibleGeolocationStatic001
    * @tc.desc       : test getAccessibleGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testGetAccessibleGeolocationStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testGetAccessibleGeolocation001', 10)
        await Utils.msSleep(2000);
        Utils.registerEvent('testGetAccessibleGeolocation001', true, 100008, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })

    /*
    * @tc.number     : SUB_WEB_GETACCESSIBLEGEOLOCATION_STATIC_0200
    * @tc.name       : testGetAccessibleGeolocationStatic002
    * @tc.desc       : test getAccessibleGeolocation
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testGetAccessibleGeolocationStatic002', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        Utils.emitEvent('testGetAccessibleGeolocation002', 10)
        await Utils.msSleep(2000);
        Utils.registerEvent('testGetAccessibleGeolocation002', true, 100009, done);
        await Utils.msSleep(3000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(2000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(2000);
        button?.click();
        await Utils.msSleep(3000);
        done();
      })

    it('SwitchNextPageStatic', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        await Utils.msSleep(1000);
        Utils.emitEvent('SwitchNextPageStatic', 10)
        await Utils.msSleep(1000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(1000);
        let button = await driver.findComponent(ON.id('GeolocationPermissionsButton'));
        await Utils.msSleep(1000);
        button?.click();
        await Utils.msSleep(4000)
        done()
      })

  })
}
