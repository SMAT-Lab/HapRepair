import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('de-DE');
calendar.setTimeZone(i18n.getTimeZone().getID());
calendar.getTime();