```arkts

/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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

import promptAction from '@ohos.promptAction'
import router from '@ohos.router'
import Pop3Command, { Pop3LoginBean } from '@ohos/node-pop3'
import MsgBean from '../MsgBean'
import GlobalObj from '../GlobalObj'

const BASE_COUNT = 1

@Entry
@Component
struct AutoLoginCommandPage {
  @State listData: Array<string> = []
  mailType: string = '@qq.com'
  @State client: Pop3Command | null = null
  @State selectMsgNum: number = -1
  isListShow: boolean = false
  @State msgList: Array<MsgBean> = []
  @State topLine: string = ''
  controller: TextInputController = new TextInputController()

  async aboutToDisappear() {
    if (this.client) {
      await this.client.QUIT()
    }
  }

  showToast(text: string, name = 'test') {
    console.log(`zdy---${name}--->${text}`)
    promptAction.showToast({
      message: text,
      duration: 2000,
      bottom: 50
    })
  }

  aboutToAppear() {
    let params = router.getParams()
    let tempParam = params as Record<string, Object>
    if (tempParam && tempParam['loginOption'] && typeof tempParam['loginOption'] === 'object') {
      let option = tempParam['loginOption'] as Pop3LoginBean;
      this.client = new Pop3Command({
        user: option?.user,
        password: option?.password,
        host: option?.host,
        port: option?.port,
        tls: option?.tls,
        timeout: option?.timeout,
        tlsOptions: option?.tlsOptions,
        servername: option?.servername
      });
    }
  }

  build() {
    Row() {
      Column() {
        Scroll() {
          Row() {