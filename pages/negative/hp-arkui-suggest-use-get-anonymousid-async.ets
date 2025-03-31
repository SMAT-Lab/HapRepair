//  建议在主线程中通过异步获取IFAA免密认证的匿名化ID。
//
// 高耗时函数处理场景下，建议优先修改。

import { ifaa } from '@kit.OnlineAuthenticationKit'

// 开发者需要按照IIFAA的TLV格式构造入参，并转换为Uint8Array参数；此处arg需要开发者替换为真实入参。
let arg = new Uint8Array([0]);
let getAnonIdResult: Uint8Array = ifaa.getAnonymousIdSync(arg);