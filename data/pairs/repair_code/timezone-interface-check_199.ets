import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('ja-JP');
calendar.setTimeZone(i18n.getTimeZone().getID());
calendar.get('zone_offset');