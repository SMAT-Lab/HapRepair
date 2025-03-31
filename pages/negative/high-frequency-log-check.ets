// Description: 不建议在高频函数中使用Hilog。
//
// 高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。
//
// 高耗时函数处理场景下，建议优先修改。

// Test.ets
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
    build() {
            Column() {
                Scroll()
                    .onScroll(() => {
                        // Avoid printing logs
                        hilog.info(1001, 'Index', 'onScroll')
                })
            }
    }
}