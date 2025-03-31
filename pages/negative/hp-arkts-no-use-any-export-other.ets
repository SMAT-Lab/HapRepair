// 避免使用export * 导出其他module中定义的类型和数据。
//
// 冷启动完成时延场景下，建议优先修改。

// 当前文件 User2.ets
// 从 Product.ets 文件中导出所有可导出的成员
export * from './hp-arkts-no-use-any-export-current';
// 从 Product.ets 文件中导出所有可导出的成员
export * as XX from './hp-arkts-no-use-any-export-current';
class User2 {
  id?: number;
  name?: string;
}