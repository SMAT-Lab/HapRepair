import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar(i18n.getSystemLocale());
calendar.setTimeZone(i18n.getTimeZone().getID()); 