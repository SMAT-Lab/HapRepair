## POP3

### Introduction

This project provides an email client that supports the POP3 protocol. It allows the client to log in to or log out of a mailbox, obtain the email list, obtain email details, and delete emails.

## How to Install

```shell
ohpm install @ohos/node-pop3 
```

For details about the OpenHarmony ohpm environment configuration, see [OpenHarmony HAR](https://gitee.com/openharmony-tpc/docs/blob/master/OpenHarmony_har_usage.en.md).

## How to Use

Note: Globally search *xxx* in the project and replace them with the actual email address and account password.

#### Importing the Library

```
import Pop3Command from '@ohos/node-pop3'
```

#### Initializing Request Parameters

```typescript
 let option = {
    host: `pop.qq.com`, // Adjust the value based on the mailbox type, for example, pop.163.com or pop.126.com.
    port: 110, // If tls is false, use port 110. If tls is true, use port 995.
    user: 'xxx@qq.com', // Enter the actual email address, which must be the same as the email address type of the host.
    password: 'xxxxxx', // Pay attention to the rules for distinguishing different email addresses. Some email addresses use the account and password, and others use the authorization code.
    timeout: 30000,
    servername: `pop.qq.com`, // The value must be the same as host.
    tls: false, // Set this parameter to false for encryption purposes.
    tlsOptions: undefined // This parameter must be set when tls is true. The type is TLSConnectOptions of the @ohos.net.socket module.
  }
```

#### Initializing the Client

```typescript
 let client = new Pop3Command(option);
```

The request process is encapsulated in the library. You can directly call commands to perform related operations without login.

#### Checking the Connection with the Server

```typescript
 await client.NOOP()
 console.log(`pop3---NOOP executed successfully --->`)
```

#### Obtaining All Unread Emails in the Mailbox

```typescript
let result = await client.STAT()
console.log(`pop3---STAT executed successfully --->${result}`)
```

If there are unread emails in the mailbox, two parts are returned, separated by spaces. The first part indicates the number of unread emails, and the second part indicates the total number of bytes of unread emails.

#### Obtaining the Email List

```typescript
let result = await client.LIST()
console.log(`pop3---LIST executed successfully --->`)
```

If there are unread emails in the mailbox, two parts are returned, separated by the comma (,), for each unread email. The odd number indicates the email sequence number, and the even number indicates the number of bytes of the email.

#### Obtaining Email Details

```typescript
 let result = awaitclient.RETR(this.selectMsgNum + '')
```

The parameter is the email sequence number. The obtained response result is encoded and needs to be parsed.

#### Obtaining the First Several Lines of an Email

```typescript
 let result = await client.TOP(this.selectMsgNum + '', parseInt(this.topLine))
```

The parameters are the email sequence number and topLine. The obtained response result is encoded and needs to be parsed.

#### Adding a DELE Flag to an Email

```typescript
 let result = await client.DELE(this.selectMsgNum + '')
```

After a DELE flag is added to an email, the email is not deleted immediately. It is deleted only after the QUIT command is called.

#### Logout

```typescript
 const [quitInfo] = await client.QUIT();
```

Log out and delete the email with the DELE tag.

### Available APIs

APIs

| Name       | Parameter               | Return Value                        | Description                                                        |
| ------------- | ------------------- | ------------------------------ | ------------------------------------------------------------ |
| listify       | str                 | string[]                       | Parses the result into a string array.                                      |
| QUIT          | N/A                 | N/A                            | Sends the QUIT command to log out and disconnect from the server.                    |
| command       | ...args             | Promise<[string,Readable]>     | Sends a client command to the email server, receives the response, and returns the response to the client.          |
| stream2String | stream              | Promise< string >              | Converts a response data stream to a string.                                    |
| DELE          | msgNumber           | Promise< string >              | Sends the DELE command to set the DELE flag for an email.                      |
| LIST          | msgNumber           | Promise< string[][]\|string[]> | Sends the LIST command to obtain the email list. The list data contains only the email sequence number and the number of email bytes.|
| UIDL          | msgNumber           | Promise< string >              | Sends the UIDL command to obtain the UID of an email.                                 |
| NOOP          | N/A                 | N/A                            | Sends the NOOP command to check the connection between the client and server.                    |
| RSET          | N/A                 | N/A                            | Sends the RSET command to clear the DELE flags of all emails.                    |
| TOP           | msgNumber, numLines| Promise< string >              | Sends the TOP command to obtain the first several lines of an email.                     |
| STAT          | N/A                 | Promise< string >              | Sends the STAT command to obtain the information about all the unread emails in the mailbox, including the number of unread emails and the total number of bytes of unread emails.|
| RETR          | msgNumber           | Promise< string >              | Sends the RETR command to obtain the content of an email.                            |
| connect       | N/A                 | string                         | Sends a request to connect to the email server.                                      |

For more details, see [Official Documentation](https://github.com/node-pop3/node-pop3/blob/master/README.md) and [Unit Test Cases](https://gitee.com/openharmony-tpc/openharmony_tpc_samples/blob/master/node-pop3/TEST.md).

### Constraints

This project has been verified in the following version:

DevEco Studio: 4.0 Release (4.0.3.413), SDK: API 10 (4.0.10.3)

The following mailbox types are supported: QQ, 163, and 126 mailboxes.

### Directory Structure

```
|-entry 
    |-ets
    |   |-entryability
    |           |-EntryAbility.ts
    |   |-pages
    |           |-Index.ets             # Used to set request parameters
    |           |-AutoLoginCommandPage.ets   # Main feature demonstration page
|-POP3   # POP3 library        
```

### How to Contribute

If you find any problem during the use, submit an [issue](https://gitee.com/openharmony-tpc/openharmony_tpc_samples/issues) or a [PR](https://gitee.com/openharmony-tpc/openharmony_tpc_samples/pulls).

### License

This project is licensed under [MIT License](https://gitee.com/openharmony-tpc/openharmony_tpc_samples/blob/master/node-pop3/LICENSE).
