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

@Component
struct PageOne99 {
  private stack: NavPathStack | undefined = undefined;

  build() {
    NavDestination() {
      Stack({alignContent: Alignment.Center}) {
        Button('push PageOne').onClick(() => {
          this.stack?.pushPath({name: 'PageOne'});
        })
      }.width('100%').height('100%')
    }.title('PageOne')
    .onReady((ctx: NavDestinationContext) => {
      this.stack = ctx.pathStack;
      console.log('HomePageInfo name: ', JSON.stringify(ctx.pathInfo.name))
      AppStorage.setOrCreate('name', ctx.pathInfo.name);
      console.log('HomePageInfo param: ', JSON.stringify(ctx.pathInfo.param))
      AppStorage.setOrCreate('param', JSON.stringify(ctx.pathInfo.param));
    })
  }
}


@Entry
@Component
struct HomePage {
  private stack: NavPathStack = new NavPathStack();

  @Builder
  PageOneBuilder() {
    PageOne99()
  }

  build() {
    // 在这里配置主页NavDestination信息
    Navigation(this.stack, { name: 'PageOne', param: {'key': 'value'} }) {
    }
    .width('100%').height('100%')
    .navDestination(this.PageOneBuilder)
    .title('Navigation')
  }
}