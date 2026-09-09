import i18n from '@ohos.i18n';

export function resolveRemoteOffset(zoneId: string): number {
  const scheduleCalendar = i18n.getCalendar(i18n.getSystemLocale());
  scheduleCalendar.setTimeZone(zoneId);
  const baseOffset: number = scheduleCalendar.get('zone_offset');
  scheduleCalendar.get('dst_offset');
  return baseOffset;
}
