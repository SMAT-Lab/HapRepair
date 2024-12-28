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

@Entry
@Component
struct TipsPage {
  @State message: string = 'Pop3 Command Description and Precautions'

  aboutToAppear() {
    this.message = this.message += '\r\n' + '\r\n' + "Command Description："
    this.message = this.message += '\r\n' + '\r\n' + "USER        username        Authentication username"
    this.message = this.message += '\r\n' + '\r\n' + "PASS         password        Authentication password authentication, If the authentication is passed, the state will transition"
    this.message = this.message += '\r\n' + '\r\n' + "APOP        name,digest    Recognize a secure method for transmitting passwords, Successful execution leads to state transition, Please refer to RFC 1321 "
    this.message = this.message += '\r\n' + '\r\n' + "STAT         Send back email statistics, such as the number of emails and the total number of bytes in emails "
    this.message = this.message += '\r\n' + '\r\n' + "UIDL n       Return the unique identifier used for the specified email. If not specified, return all "
    this.message = this.message += '\r\n' + '\r\n' + "LIST n       Return the size of the specified email, etc. If not specified, return all"
    this.message = this.message += '\r\n' + '\r\n' + "RETR n     Return all text of the email, if not specified, return all"
    this.message = this.message += '\r\n' + '\r\n' + "DELE n      Delete, the QUIT command is executed to truly delete"
    this.message = this.message += '\r\n' + '\r\n' + "RSET         Revoke all DELE commands"
    this.message = this.message += '\r\n' + '\r\n' + "TOP n,m     Return the first m lines of email n, where m must be a natural number"
    this.message = this.message += '\r\n' + '\r\n' + "NOOP        Return a positive response"
    this.message = this.message += '\r\n' + '\r\n' + "QUIT          End Session"
    this.message = this.message += '\r\n' + '\r\n' + "Note："
    this.message = this.message += '\r\n' + '\r\n' + "USER Used for sending email accounts"
    this.message = this.message += '\r\n' + '\r\n' + "PASS Used to send email passwords, some emails require authorization codes, such as QQ 163"
    this.message = this.message += '\r\n' + '\r\n' + "STAT Used to return the number and size of all emails in the mailbox. Currently, QQ mailbox only returns the email statistics for that day"
    this.message = this.message += '\r\n' + '\r\n' + "LIST Used to return the email sequence number and size in the mailbox, the sequence number for not sending emails is the sequence number and size of all emails returned, and the sequence number for sending a certain email is the information of the current email"
    this.message = this.message += '\r\n' + '\r\n' + "DELE Used to delete emails, but requires re publishing QUIT after DELE is successful for it to take effect. Currently, the QQ email being tested can take effect by sending DELE"
    this.message = this.message += '\r\n' + '\r\n' + "RSET Used to undo the deletion of emails, but it needs to be executed after DELE is successful, and then QUIT will be published for it to take effect. Currently, the QQ mailbox being tested has directly deleted emails sent with DELE"
    this.message = this.message += '\r\n' + '\r\n' + "RETR Used to obtain email details, the returned data is encrypted and encoded, and needs to be decoded to display normal content. When there is too much data, it will not be displayed on the text and requires partial data in the stage"
    this.message = this.message += '\r\n' + '\r\n' + "TOP The first few lines used to retrieve email information"
    this.message = this.message += '\r\n' + '\r\n' + "NOOP Used to confirm whether the connection with the server is normal, similar to ping/ong in WebSocket"
    this.message = this.message += '\r\n' + '\r\n' + "QUIT Log out and start performing deletion and other operations"
  }

  build() {
    Row() {
      Column() {
        Scroll() {
          Text(this.message)
            .fontSize(20)
            .textAlign(TextAlign.Start)
            .padding(20)
        }
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
  }
}