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
export interface AnimateCallback {
  finish: (isPush: boolean, isExit: boolean) => void;
  start: (isPush: boolean, isExit: boolean) => void;
  timeout: number;
}
const customTransitionMap: Map<string, AnimateCallback> = new Map()
export class CustomTransition {
  private constructor() {

  }

  static delegate = new CustomTransition();

  static getInstance() {
    return this.delegate;
  }

  registerNavParam(name: string, startCallback: (operation: boolean, isExit: boolean) => void,
                   endCallback:(operation: boolean, isExit: boolean) => void, timeout: number): void {

    if (customTransitionMap.has(name)) {
      let param = customTransitionMap.get(name);
      param.start = startCallback;
      param.finish = endCallback;
      param.timeout = timeout;
      return;
    }
    let params: AnimateCallback = {timeout: timeout, start: startCallback, finish: endCallback};
    customTransitionMap.set(name, params);
  }

  unRegisterNavParam(name: string): void {
    customTransitionMap.delete(name);
  }

  getAnimateParam(name: string): AnimateCallback {
    let result: AnimateCallback = {
      start: customTransitionMap.get(name).start,
      finish: customTransitionMap.get(name).finish,
      timeout: customTransitionMap.get(name).timeout
    };
    return result;
  }
}