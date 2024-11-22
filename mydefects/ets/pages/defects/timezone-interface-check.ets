import i18n from '@ohos.i18n';

// 手动设置时区，不通过标准接口获取时间信息
let timeZone = '123';
let calendar = i18n.getCalendar(i18n.getSystemLocale());
calendar.setTimeZone(timeZone);

// 仅获取时区偏移，但未处理DST（夏令时）
let zoneOffset = calendar.get('zone_offset');  // 获取时区偏移，但缺少dst_offset获取
// 缺少对dst_offset（夏令时偏移）的获取
