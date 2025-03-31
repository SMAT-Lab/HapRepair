import { BusinessError } from '@ohos.base';

try {
  // 本地resources中配置的color资源
  this.context.resourceManager.getColorSync($r('app.color.test').id);
} catch (error) {
  let code = (error as BusinessError).code;
  let message = (error as BusinessError).message;
  console.error(`getColorSync failed, error code: ${code}, message: ${message}.`);
}