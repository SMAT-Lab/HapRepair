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
export type Callback = (message: ESObject) => void

export class MessageManager {
  callback: Callback
  callbacks: Map<string, Callback> = new Map()

  constructor() {

  }

  registerCallback(callback: Callback, type?: string) {
    if (type == undefined) {
      console.error('registerCallback callback set')
      this.callback = callback
      return
    }
    this.callbacks.set(type, callback)
  }

  notify(message: any, type?: string) {
    if (type == undefined) {
      this.callback(message)
      return
    }

    let tmpCallback: Callback = this.callbacks.get(type)
    if (tmpCallback === undefined) {
      console.error('callbacks has no callback for type ' + type)
      return
    }
    tmpCallback(message)
  }

  clear() {
    this.callbacks.clear()
    this.callback = null
  }
}