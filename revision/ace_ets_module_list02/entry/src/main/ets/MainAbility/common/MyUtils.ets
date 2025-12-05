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

import events_emitter from '@ohos.events.emitter';
import RectValue from './Rect'

export default class Utils {
  static rect_left: number;
  static rect_top: number;
  static rect_right: number;
  static rect_bottom: number;
  static rect_value: RectValue;

  static sleep(time: number) {
    return new Promise((resolve: (value: string) => void) => {
      setTimeout(() => {
        resolve("ok")
      }, time)
    }).then(() => {
      console.info(`sleep ${time} over...`)
    })
  }

  static sendEvent(message: string, eventId: number) {
    let eventData: events_emitter.EventData = {
      data: {
        "msg": message
      }
    }
    let event: events_emitter.InnerEvent = {
      eventId: eventId,
      priority: events_emitter.EventPriority.LOW
    }
    events_emitter.emit(event, eventData)
  }

  //获取组件所占矩形区域坐标
  static getComponentRect(key: string): RectValue {
    let strJson: string = getInspectorByKey(key);
    let rectInfo: Array<string> = JSON.parse('[' + JSON.parse(strJson).$rect + ']');
    Utils.rect_left = JSON.parse('[' + rectInfo[0] + ']')[0];
    Utils.rect_top = JSON.parse('[' + rectInfo[0] + ']')[1];
    Utils.rect_right = JSON.parse('[' + rectInfo[1] + ']')[0];
    Utils.rect_bottom = JSON.parse('[' + rectInfo[1] + ']')[1];
    return Utils.rect_value = {
      left: Utils.rect_left, top: Utils.rect_top, right: Utils.rect_right, bottom: Utils.rect_bottom
    }
  }
}