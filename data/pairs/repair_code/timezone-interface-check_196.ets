import i18n from '@ohos.i18n';

function getLocalTime() {
    let calendar = i18n.getCalendar(i18n.getSystemLocale());
    return calendar.getTime();
}