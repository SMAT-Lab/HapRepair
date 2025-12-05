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

import router from '@ohos.router';
import { AttrsManager } from './AttrsManager';


export class RectValue {
  left: number = 0;
  top: number = 0;
  right: number = 0;
  bottom: number = 0;
}

export default class Utils {
  public static instance: Utils;

  static sleep(time: number) {
    return new Promise<string>((resolve, reject) => {
      setTimeout(() => {
        resolve("ok");
      }, time);
    }).then(() => {
      console.info(`sleep ${time} over...`);
    });
  }

  static getStringFromFile(fileName: string): Promise<string> {
    return globalThis.abilityContext
      .resourceManager
      .getRawFileContent(fileName)
      .then((data: ESObject) => {
        return new Promise<ESObject>((resolve, reject) => {
          if (data) {
            let result = '';
            for (let i = 0; i < data.length; i++) {
              result += String.fromCharCode(data[i]);
            }
            resolve(result);
          } else {
            reject('')
          }
        })
      })
  }

  static async pushPage(obj: ESObject) {
    try {
      let pages = router.getState();
      let pageParams: ESObject = router.getParams();
      console.info('get ' + obj.pageName + ' state pages: ' + JSON.stringify(pages));
      if (pageParams && pageParams['view'] && obj.targetView != pageParams['view']['targetView']) {
        console.info('get ' + obj.pageName + ' params: ' + JSON.stringify(pageParams));
        console.info('pageParams.view.targetView: ' + pageParams['view']['targetView']);
        console.info('targetView: ' + obj.targetView);
        await Utils.back();
        pages = router.getState();
      }

      if (!(obj.pageName == pages.name)) {
        console.info('get ' + obj.pageName + ' state pages.name:' + JSON.stringify(pages.name));
        let result = await router.pushUrl({
          url: obj.pageUrl,
          params: {
            view: obj
          }
        });
        await Utils.sleep(2000);
        console.info('push ' + obj.pageName + ' page result:' + JSON.stringify(result));
      }
    } catch (err) {
      console.error('push ' + obj.pageName + ' page error:' + err);
    }
  }

  static async back() {
    if (!('Index' == router.getState().name)) {
      router.back()
    }
    AttrsManager.unRegisterDataChange();
    await Utils.sleep(1000);
    router.clear();
  }

  static async backOnly() {
    router.back()
    await Utils.sleep(1000);
  }

  static getComponentRect(key: string): RectValue {
    let strJson = getInspectorByKey(key);
    console.info('[getInspectorByKey] key is: ' + key);
    console.info("[getInspectorByKey] rectInfo is: " + JSON.parse('[' + JSON.parse(strJson).$rect + ']'));
    return Utils.getComponentRectByObj(JSON.parse(strJson));
  }

  static getComponentRectByObj(obj: ESObject): RectValue {
    console.info('[getInspectorByKey] current component is: ' + obj.$type);
    let rectInfo: ESObject = JSON.parse('[' + obj.$rect + ']');
    let rect_value: RectValue = {
      left: Number(JSON.parse('[' + rectInfo[0] + ']')[0]),
      top: Number(JSON.parse('[' + rectInfo[0] + ']')[1]),
      right: Number(JSON.parse('[' + rectInfo[1] + ']')[0]),
      bottom: Number(JSON.parse('[' + rectInfo[1] + ']')[1])
    };
    console.info(JSON.stringify(rect_value));
    return rect_value;
  }

  static getComponentByKey(componentKey: string): ESObject {
    let strJson = getInspectorByKey(componentKey);
    console.info('[' + componentKey + '] obj is: ' + JSON.stringify(JSON.parse(strJson)));
    return JSON.parse(strJson);
  }

  static getRectHorizontalCenterX(rect: RectValue) {
    (rect.right - rect.left) / 2 + rect.left;
  }

  static getRectVerticalCenterY(rect: RectValue) {
    (rect.bottom - rect.top) / 2 + rect.top;
  }

  static requestFocus(id: string): boolean {
    console.info('requestFocus id : ' + id);
    let result = focusControl.requestFocus(id);
    console.info('requestFocus result : ' + result);
    return result;
  }

  static performClick(id: string): boolean {
    console.info('performClick id : ' + id);
    let result = sendEventByKey(id, 10, '');
    console.info('performClick result : ' + result);
    return result
  }

  static async performClickPoint(x: number, y: number) {
    Utils.performTouch(x, y, x, y, TouchType.Down);
    await Utils.sleep(100);
    Utils.performTouch(x, y, x, y, TouchType.Up);
  }

  static performTouchDownUp(id: string): boolean {
    let down = Utils.performTouchComponentCenter(id, TouchType.Down)
    if (down) {
      let up = Utils.performTouchComponentCenter(id, TouchType.Up)
      return up
    }

    return false;
  }

  static performTouchComponentCenter(id: string, type: TouchType): boolean {
    console.info('performTouchComponent id : ' + id);
    return Utils.performTouchRectCenter(0, 0, Utils.getComponentRect(id), type);
  }

  static performTouchRectCenter(rootX: number, rootY: number, rect: RectValue, type: TouchType): boolean {
    console.info('performTouchRectCenter parentX : ' + rootX);
    console.info('performTouchRectCenter parentY : ' + rootY);
    console.info('performTouchRectCenter rect.left : ' + rect.left);
    console.info('performTouchRectCenter rect.top : ' + rect.top);
    console.info('performTouchRectCenter rect.right : ' + rect.right);
    console.info('performTouchRectCenter rect.bottom : ' + rect.bottom);

    let screenX = (rect.right + rect.left) * 0.5;
    let screenY = (rect.bottom + rect.top) * 0.5;
    let x = (rect.right + rect.left) * 0.5 - rootX;
    let y = (rect.bottom + rect.top) * 0.5 - rootY;

    return Utils.performTouch(screenX, screenY, x, y, type);
  }

  static performTouch(screenX: number, screenY: number, x: number, y: number, type: TouchType): boolean {
    console.info('performTouch type : ' + type);
    console.info('performTouch screenX : ' + screenX);
    console.info('performTouch screenY : ' + screenY);
    console.info('performTouch x : ' + x);
    console.info('performTouch y : ' + y);

    let result = sendTouchEvent({
      windowX: screenX,
      windowY: screenX,
      displayX: screenX,
      displayY: screenX,
      type: type,
      id: 0,
      screenX: screenX,
      screenY: screenX,
      x: x,
      y: y,
    })

    console.info('performTouch result : ' + result);

    return result;
  }

  static performKeyEventDownUp(keyCode: number, keyText: string): boolean {
    let down = Utils.performKeyEvent(KeyType.Down, keyCode, keyText)
    if (down) {
      let up = Utils.performKeyEvent(KeyType.Up, keyCode, keyText)
      return up
    }

    return false;
  }

  static performKeyEvent(type: KeyType, keyCode: number, keyText: string): boolean {
    console.info('performKeyEvent type : ' + type);
    console.info('performKeyEvent keyCode : ' + keyCode);
    console.info('performKeyEvent keyText : ' + keyText);

    let result = sendKeyEvent({
      type: type,
      keyCode: keyCode,
      keyText: keyText,
      intentionCode: -1,
      keySource: 1,
      deviceId: 0,
      metaKey: 0,
      timestamp: new Date().getTime(),
      stopPropagation: () => {
      },
       getModifierKeyState:(keys: Array<string>):boolean => {
              console.info("the key 0 is :" + keys[0]);
              return true;
        }
    })
    console.info('performKeyEvent result : ' + result);
    return result
  }

  static performMouseEvent(id: string, button: MouseButton, action: MouseAction): boolean {
    console.info('performMouseEvent id : ' + id);
    console.info('performMouseEvent button : ' + button);
    console.info('performMouseEvent action : ' + action);

    let targetComponentRect = Utils.getComponentRect(id);
    let screenX = (targetComponentRect.right + targetComponentRect.left) * 0.5;
    let screenY = (targetComponentRect.bottom + targetComponentRect.top) * 0.5;
    let x = (targetComponentRect.right + targetComponentRect.left) * 0.5;
    let y = (targetComponentRect.bottom + targetComponentRect.top) * 0.5;
    let width = targetComponentRect.right - targetComponentRect.left;
    let height = targetComponentRect.bottom - targetComponentRect.top;
    console.info('performMouseEvent screenX : ' + screenX);
    console.info('performMouseEvent screenY : ' + screenY);
    console.info('performMouseEvent x : ' + x);
    console.info('performMouseEvent y : ' + y);
    console.info('performMouseEvent width : ' + y);
    console.info('performMouseEvent height : ' + y);

    let result = sendMouseEvent({
      windowX: screenX,
      windowY: screenX,
      displayX: screenX,
      displayY: screenX,
      screenX: screenX,
      screenY: screenX,
      x: x,
      y: y,
      button: button,
      action: action,
      timestamp: new Date().getTime(),
      target: {
        area: {
          width: width,
          height: height,
          position: {
            x: 0, y: 0
          },
          globalPosition: {
            x: 0, y: 0
          },
        }
      },
      source: SourceType.Mouse,
      tiltX: 0,
      tiltY: 0,
      sourceTool: 1,
      pressure: 0,
      stopPropagation: () => {
      },
       getModifierKeyState:(keys: Array<string>):boolean => {
              console.info("the key 0 is :" + keys[0]);
              return true;
        }
    })

    console.info('performMouseEvent result : ' + result);

    return result;
  }
}