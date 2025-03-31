// 建议在使用API getColorSync和getStringSync时建议使用带id版本。
//
// 高耗时函数处理场景下，建议优先修改。

import { BusinessError } from '@ohos.base';

try {
  // 本地resources中配置的color资源
  this.context.resourceManager.getColorSync($r('app.color.test'));
} catch (error) {
  let code = (error as BusinessError).code;
  let message = (error as BusinessError).message;
  console.error(`getColorSync failed, error code: ${code}, message: ${message}.`);
}