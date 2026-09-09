# @performance/timezone-interface-check

Static repair references: 8. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0c7f5f95bf5c7bd1`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('ja-JP');
calendar.setTimeZone('short');
calendar.get('zone_offset');
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('ja-JP');
calendar.setTimeZone(i18n.getTimeZone().getID());
calendar.get('zone_offset');
```

### Rationale

使用Date对象获取时间项目，没有使用统一标准的i18n.Calendar。

## Example 2: `pair_1162e64dd3adf441`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
let regionTime = new Intl.DateTimeFormat('de-DE').format(new Date());
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('de-DE');
calendar.setTimeZone(i18n.getTimeZone().getID());
calendar.getTime();
```

### Rationale

未采用i18n.Calendar接口获取时间信息。

## Example 3: `pair_1c161a594bdc4858`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
var userLocale = 'en-US';
var dateTime = new Date().toLocaleString(userLocale);
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('en-US');
calendar.getTime()
```

### Rationale

应使用i18n.Calendar接口而不是直接使用Date对象。

## Example 4: `pair_66dd25a470028719`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
let timeZoneName = new Date().toLocaleString('ja-JP', { timeZoneName: 'short' });
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('ja-JP');
calendar.setTimeZone('short');
calendar.get('zone_offset');
```

### Rationale

使用Date对象获取时间项目，没有使用统一标准的i18n.Calendar。

## Example 5: `pair_686225099cd465a8`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
function getLocalTime() {
    return new Date().toLocaleString();
}
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

function getLocalTime() {
    let calendar = i18n.getCalendar(i18n.getSystemLocale());
    return calendar.getTime();
}
```

### Rationale

直接使用Date对象的localeString方法，没有使用i18n.Calendar的标准化方法。

## Example 6: `pair_8ba9faf44f8049d6`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
const formattedTime = new Date().toLocaleTimeString('fr-FR');
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar('fr-FR');
calendar.getTime();
```

### Rationale

直接通过Date对象获取时间，没有使用i18n.Calendar接口。

## Example 7: `pair_9a1b0299f94680d0`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
import systemDateTime from '@ohos.systemDateTime';
systemDateTime.setTimezone();
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let calendar = i18n.getCalendar(i18n.getSystemLocale());
calendar.setTimeZone(i18n.getTimeZone().getID());
```

### Rationale

建议使用统一标准的i18n.Calendar接口获取时间时区相关信息

## Example 8: `pair_d43c8a2f6755a9ea`

在获取非本地时间时，建议使用统一标准的i18n.Calendar接口获取时间时区相关信息。

### Triggering pattern

```arkts
import i18n from '@ohos.i18n';

let timeZone1 = '123';
let calendar1 = i18n.getCalendar(i18n.getSystemLocale());
calendar1.setTimeZone(timeZone1);
calendar1.get('zone_offset');
```

### Repair pattern

```arkts
import i18n from '@ohos.i18n';

let timeZone1 = '123';
let calendar1 = i18n.getCalendar(i18n.getSystemLocale());
calendar1.setTimeZone(timeZone1);
calendar1.get('zone_offset'); 
calendar1.get('dst_offset');
```

### Rationale

缺少获取dst_offset
