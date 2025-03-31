import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('ja-JP');
calendar.setTimeZone('short');
calendar.get('zone_offset');