// 在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

import i18n from '@ohos.i18n';

let timeZone1 = '123';
let calendar1 = i18n.getCalendar(i18n.getSystemLocale());
calendar1.setTimeZone(timeZone1);
//告警，缺少获取dst_offset
calendar1.get('zone_offset');
//calendar1.get('dst_offset');