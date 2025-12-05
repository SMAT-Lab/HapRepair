/*
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

import hilog from '@ohos.hilog';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { BusinessError } from '@ohos.base';

export default function stageRouterTest() {
  describe('stageRouterTest', () => {
    // Defines a test suite. Two parameters are supported: test suite name and test suite function.
    beforeAll(() => {
      // Presets an action, which is performed only once before all test cases of the test suite start.
      // This API supports only one parameter: preset action function.
    })
    beforeEach(async (done: Function) => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/stageRouter',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get stageRouter state success " + JSON.stringify(pages));
        if (!("stageRouter" == pages.name)) {
          console.info("get stageRouter state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          console.info("push stageRouter page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push stageRouter page error: " + err);
      }
      done()
    });
    afterEach(() => {
      // Presets a clear action, which is performed after each unit test case ends.
      // The number of execution times is the same as the number of test cases defined by **it**.
      // This API supports only one parameter: clear action function.
    })
    afterAll(() => {
      // Presets a clear action, which is performed after all test cases of the test suite end.
      // This API supports only one parameter: clear action function.
    })

    /**
     * @tc.number testrouterstage0001
     * @tc.name testrouterstage0001
     * @tc.desc testrouterstage0001
     */
    it('testrouterstage0001', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0001 START');
      let result: ESObject;
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/pushUrlasync.ets',
        params: {
          data1: 'message',
          data2: {
            data3: [123, 456, 789]
          }
        }
      }
      try {
        result = router.pushUrl(options)
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0001 END');
      done();
    });

    /**
     * @tc.number testrouterstage0002
     * @tc.name testrouterstage0002
     * @tc.desc testrouterstage0002
     */
    it('testrouterstage0002', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0002 START');
      let result: ESObject;
      try {
        result = router.pushUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }, (err: BusinessError) => {
          if (err) {
            console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('pushUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0002 END');
      done();
    });

    /**
     * @tc.number testrouterstage0003
     * @tc.name testrouterstage0003
     * @tc.desc testrouterstage0003
     */
    it('testrouterstage0003', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0003 START');
      let result: ESObject;
      try {
        result = router.pushUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }, router.RouterMode.Standard)
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0003 END');
      done();
    });

    /**
     * @tc.number testrouterstage0004
     * @tc.name testrouterstage0004
     * @tc.desc testrouterstage0004
     */
    it('testrouterstage0004', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0004 START');
      let result: ESObject;
      try {
        result = router.pushUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }, router.RouterMode.Standard)
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0004 END');
      done();
    });

    /**
     * @tc.number testrouterstage0005
     * @tc.name testrouterstage0005
     * @tc.desc testrouterstage0005
     */
    it('testrouterstage0005', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0005 START');
      let result: ESObject;
      try {
        result = router.replaceUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message'
          }
        })
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0005 END');
      done();
    });

    /**
     * @tc.number testrouterstage0006
     * @tc.name testrouterstage0006
     * @tc.desc testrouterstage0006
     */
    it('testrouterstage0006', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0006 START');
      let result: ESObject;
      try {
        result = router.replaceUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message'
          }
        }, (err: BusinessError) => {
          if (err) {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('replaceUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0006 END');
      done();
    });

    /**
     * @tc.number testrouterstage0007
     * @tc.name testrouterstage0007
     * @tc.desc testrouterstage0007
     */
    it('testrouterstage0007', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0007 START');
      let result: ESObject;
      try {
        result = router.replaceUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message'
          }
        }, router.RouterMode.Standard)
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0007 END');
      done();
    });

    /**
     * @tc.number testrouterstage0008
     * @tc.name testrouterstage0008
     * @tc.desc testrouterstage0008
     */
    it('testrouterstage0008', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0008 START');
      let result: ESObject;
      try {
        result = router.replaceUrl({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets',
          params: {
            data1: 'message'
          }
        }, router.RouterMode.Standard, (err: BusinessError) => {
          if (err) {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('replaceUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0008 END');
      done();
    });

    /**
     * @tc.number testrouterstage0009
     * @tc.name testrouterstage0009
     * @tc.desc testrouterstage0009
     */
    it('testrouterstage0009', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0009 START');
      let result: ESObject;
      try {
        result = router.back({
          url: 'MainAbility/pages/Router2/pushUrlasync.ets'
        })
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0009 END');
      done();
    });

    /**
     * @tc.number testrouterstage0010
     * @tc.name testrouterstage0010
     * @tc.desc testrouterstage0010
     */
    it('testrouterstage0010', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0010 START');
      let result: ESObject;
      try {
        result = router.clear()
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0010 END');
      done();
    });

    /**
     * @tc.number testrouterstage0011
     * @tc.name testrouterstage0011
     * @tc.desc testrouterstage0011
     */
    it('testrouterstage0011', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0011 START');
      let result: ESObject;
      try {
        result = router.getLength()
        expect(typeof (result)).assertEqual('string');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0011 END');
      done();
    });

    /**
     * @tc.number testrouterstage0012
     * @tc.name testrouterstage0012
     * @tc.desc testrouterstage0012
     */
    it('testrouterstage0012', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0012 START');
      let result: ESObject;
      try {
        result = router.getState()
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0012 END');
      done();
    });

    /**
     * @tc.number testrouterstage0013
     * @tc.name testrouterstage0013
     * @tc.desc testrouterstage0013
     */
    it('testrouterstage0013', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0013 START');
      let result: ESObject;
      try {
        result = router.showAlertBeforeBackPage({
          message: 'Message Info'
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0013 END');
      done();
    });

    /**
     * @tc.number testrouterstage0014
     * @tc.name testrouterstage0014
     * @tc.desc testrouterstage0014
     */
    it('testrouterstage0014', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0014 START');
      let result: ESObject;
      try {
        result = router.hideAlertBeforeBackPage()
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0014 END');
      done();
    });

    /**
     * @tc.number testrouterstage0015
     * @tc.name testrouterstage0015
     * @tc.desc testrouterstage0015
     */
    it('testrouterstage0015', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0015 START');
      let result: ESObject;
      try {
        result = router.getParams()
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0015 END');
      done();
    });

    /**
     * @tc.number testrouterstage0016
     * @tc.name testrouterstage0016
     * @tc.desc testrouterstage0016
     */
    it('testrouterstage0016', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0016 START');
      let result: ESObject;
      try {
        result = router.pushNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        })
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0016 END');
      done();
    });

    /**
     * @tc.number testrouterstage0017
     * @tc.name testrouterstage0017
     * @tc.desc testrouterstage0017
     */
    it('testrouterstage0017', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0017 START');
      let result: ESObject;
      try {
        result = router.pushNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }, (err: BusinessError) => {
          if (err) {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('replaceUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0017 END');
      done();
    });

    /**
     * @tc.number testrouterstage0018
     * @tc.name testrouterstage0018
     * @tc.desc testrouterstage0018
     */
    it('testrouterstage0018', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0018 START');
      let result: ESObject;
      try {
        result = router.pushNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }, router.RouterMode.Standard)
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`replaceUrl failed, code is ${(err as BusinessError).code}, message is ${(err as BusinessError).message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0018 END');
      done();
    });

    /**
     * @tc.number testrouterstage0019
     * @tc.name testrouterstage0019
     * @tc.desc testrouterstage0019
     */
    it('testrouterstage0019', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0019 START');
      let result: ESObject;
      try {
        result = router.pushNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }, router.RouterMode.Standard, (err: BusinessError) => {
          if (err) {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('replaceUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0019 END');
      done();
    });

    /**
     * @tc.number testrouterstage0020
     * @tc.name testrouterstage0020
     * @tc.desc testrouterstage0020
     */
    it('testrouterstage0020', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0020 START');
      let result: ESObject;
      try {
        result = router.replaceNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message'
          }
        })
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0020 END');
      done();
    });

    /**
     * @tc.number testrouterstage0021
     * @tc.name testrouterstage0021
     * @tc.desc testrouterstage0021
     */
    it('testrouterstage0021', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0021 START');
      let result: ESObject;
      try {
        result = router.replaceNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message'
          }
        }, (err: BusinessError) => {
          if (err) {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('replaceUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0021 END');
      done();
    });

    /**
     * @tc.number testrouterstage0022
     * @tc.name testrouterstage0022
     * @tc.desc testrouterstage0022
     */
    it('testrouterstage0022', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0022 START');
      let result: ESObject;
      try {
        result = router.replaceNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message'
          }
        }, router.RouterMode.Standard)
          .then(() => {
          })
          .catch((err: BusinessError) => {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        expect(typeof (result)).assertEqual('object');
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0022 END');
      done();
    });

    /**
     * @tc.number testrouterstage0023
     * @tc.name testrouterstage0023
     * @tc.desc testrouterstage0023
     */
    it('testrouterstage0023', Level.LEVEL2, async (done: Function) => {
      let router: ESObject = globalThis.uiContent.getRouter();
      console.info('testrouterstage0023 START');
      let result: ESObject;
      try {
        result = router.replaceNamedRoute({
          name: 'pushUrl',
          params: {
            data1: 'message'
          }
        }, router.RouterMode.Standard, (err: BusinessError) => {
          if (err) {
            console.error(`replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            return;
          }
          console.info('replaceUrl success');
        });
        expect(result).assertUndefined();
        console.info("push button page success " + JSON.stringify(result));
      } catch (err) {
        console.error("push button page error " + JSON.stringify(result));
      }
      console.info('testrouterstage0023 END');
      done();
    });


  })
}