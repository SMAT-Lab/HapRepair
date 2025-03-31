// Test.ets
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
    build() {
            Column() {
                Scroll()
                    .onScroll(() => {
                        hilog.info(1001, 'Index', 'onScroll') // Avoid printing logs
                })
            }
    }
}