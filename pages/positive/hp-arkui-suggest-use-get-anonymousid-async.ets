import { ifaa } from '@kit.OnlineAuthenticationKit'
import { BusinessError } from '@kit.BasicServicesKit';

// 开发者需要按照IIFAA的TLV格式构造入参，并转换为Uint8Array参数；此处arg需要开发者替换为真实入参。
let arg = new Uint8Array([0]);
let getAnonIdPromise: Promise<Uint8Array> = ifaa.getAnonymousId(arg);
getAnonIdPromise.then(result => {
  console.info("Succeeded in doing getAnonymousId.");
  // 开发者处理result
}).catch((err: BusinessError) => {
  console.error(`Failed to call getAnonymousId. Code: ${err.code}, message: ${err.message}`);
});