// 系统插件
import { appTasks } from '@ohos/hvigor-ohos-plugin';
// 1. 导入在线签名插件
import { onlineSignPlugin } from '@ohos/hvigor-ohos-online-sign-plugin';
import type { OnlineSignOptions } from '@ohos/hvigor-ohos-online-sign-plugin';

// 2. 配置签名参数
const signOptions: OnlineSignOptions = {
  profile: 'hw_sign/debug.p7b',           // 签名材料
  keyAlias: 'HOS Application Provision Debug V2',
  hapSignToolFile: `${process.env.HAP_SIGN_TOOL ??
    'hw_sign/hap-sign-tool.jar'}`,      // 签名工具hap-sign-tool.jar的路径
  username: `${process.env.W3_ACCOUNT}`,  // 环境变量中需要配置用户名和密码
  password: `${process.env.W3_PASSWORD}`,
  enableOnlineSign: true                  // 是否启用在线签名
}

// hvigorfile 导出范式
export default {
  system: appTasks,
  plugins:[
    // 3. 应用插件
    onlineSignPlugin(signOptions)
  ]
}