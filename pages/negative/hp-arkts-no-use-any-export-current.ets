// 避免使用export * 导出当前module中定义的类型和数据。
//
// 冷启动完成时延场景下，建议优先修改。

class User {
  id?: number;
  name?: string;
}
// 当前文件 User.ets
export * from './hp-arkts-no-use-any-export-current';
// 当前文件 User.ets
export * as XX from './hp-arkts-no-use-any-export-current';