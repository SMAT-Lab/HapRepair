import i18n from '@ohos.i18n';

let timeZone1 = '123';
let calendar1 = i18n.getCalendar(i18n.getSystemLocale());
calendar1.setTimeZone(timeZone1);
calendar1.get('zone_offset'); 
calendar1.get('dst_offset');