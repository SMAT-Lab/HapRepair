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
import windowAnimationManager from '@ohos.animation.windowAnimationManager';
import WindowAnimationControllerImpl from './WindowAnimationControllerImpl';
import matrix4 from '@ohos.matrix4'

let matrix = matrix4.init([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

@Entry
@Component
export default struct comSizeAbilityTest_21 {
  @State target: WindowAnimationTarget | undefined = undefined // 通过windowAnimationManager获取

  aboutToAppear(): void {
    let controller: WindowAnimationControllerImpl = new WindowAnimationControllerImpl();
    windowAnimationManager.setController(controller);
    controller.onStartAppFromLauncher = (startingWindowTarget: WindowAnimationTarget,
      finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback) => {
      console.log(`RemoteWindowExample: remote window animaion onStartAppFromLauncher`);
      this.target = startingWindowTarget;
      finishedCallback.onAnimationFinish();
    }
    controller.onStartAppFromRecent = (startingWindowTarget: WindowAnimationTarget,
      finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback) => {
      console.log(`RemoteWindowExample: remote window animaion onStartAppFromRecent`);
      this.target = startingWindowTarget;
      finishedCallback.onAnimationFinish();
    }
    controller.onStartAppFromOther = (startingWindowTarget: WindowAnimationTarget,
      finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback) => {
      console.log(`RemoteWindowExample: remote window animaion onStartAppFromOther`);
      this.target = startingWindowTarget;
      finishedCallback.onAnimationFinish();
    }
    controller.onAppTransition = (fromWindowTarget: WindowAnimationTarget, toWindowTarget: WindowAnimationTarget,
      finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback) => {
      console.log(`RemoteWindowExample: remote window animaion onAppTransition`);
      this.target = toWindowTarget;
      finishedCallback.onAnimationFinish();
    }
    controller.onMinimizeWindow = (minimizingWindowTarget: WindowAnimationTarget,
      finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback) => {
      console.log(`RemoteWindowExample: remote window animaion onMinimizeWindow`);
      this.target = minimizingWindowTarget;
      finishedCallback.onAnimationFinish();
    }
    controller.onCloseWindow = (closingWindowTarget: WindowAnimationTarget,
      finishedCallback: windowAnimationManager.WindowAnimationFinishedCallback) => {
      console.log(`RemoteWindowExample: remote window animaion onCloseWindow`);
      this.target = closingWindowTarget;
      finishedCallback.onAnimationFinish();
    }
  }

  build() {
    Column() {
      if (this.target) {
        RemoteWindow(this.target)
          .size({ width: 180, height: 130 })
          .scale({ x: 0.5, y: 0.5 })// 仅用于可见效果的演示目的，正常使用须 .scale({ x: 1, y: 1 })
          .position({ x: px2vp(this.target?.windowBounds.left), y: px2vp(this.target?.windowBounds.top) })
          .width(px2vp(this.target?.windowBounds.width))
          .height(px2vp(this.target?.windowBounds.height))
          .key('RemoteWindow_Size')
          .translate({ x: 1, y: 1, z: 1 })
          .scale({
            x: 1,
            y: 1,
            z: 1,
            centerX: 1,
            centerY: 1
          })
          .rotate({
            x: 1,
            y: 1,
            z: 1,
            angle: 0,
            centerX: 1,
            centerY: 1
          })
          .transform(matrix)
      }
    }.key('RemoteWindow_Father')
  }
}