# 为什么需要 LLM：三类“重构型”缺陷的 case（TOSEM 扩展版素材）

目标：用 3 个具体 CodeLinter 规则的真实代码片段说明——有些缺陷修复并非“局部替换/固定模板”，而是涉及组件抽取、接口设计、状态提升与回调接线等设计决策；这类任务很难靠确定性规则覆盖，因此需要 LLM 参与生成式重构，并配合编译/测试/静态检查做验证闭环。

## Case 1: `@performance/hp-arkui-use-onAnimationStart-for-swiper-preload`（Swiper 预加载回调接线）

**规则意图（直观理解）**：Swiper 的预加载建议通过 `onAnimationStart`（或等价机制）在合适时机触发，避免在滚动/切换过程中出现卡顿或数据准备不及时。

**代表性代码片段**（缺少 `onAnimationStart`，但已有 `onChange` / 数据源增删等逻辑）

```ts
@Entry
@Component
struct maintainContentPositionSwiper {
  private swiperController: SwiperController = new SwiperController()
  private data: MyDataSource = new MyDataSource()
  @State maintainContentPosition: boolean = false
  @State index: number = 5

  build() {
    Column() {
      // ...按钮触发数据源 add/delete ...
      Text('Swiper: ' + this.index).id('swiper_maintain_index')
      Swiper(this.swiperController) {
        LazyForEach(this.data, (item: string, index) => {
          SwiperItemComponent({
            itemText: item.toString()
          }).reuseId(item.toString())
        }, (item: string) => item).cachedCount(4)
      }
      .index(this.index)
      .maintainVisibleContentPosition(this.maintainContentPosition)
      .onChange((index) => {
        this.index = index
        console.error('Swiper onChange ' + index)
      })
    }.width('100%')
  }
}
```

**为什么很难用固定规则“安全自动修”**（需要 LLM 的点）
- 需要理解“预加载”应该加载什么（数据源？图片？子组件状态？）以及触发时机（动画开始/结束、index 变化前后）。
- 需要选择把逻辑接到哪一层：Swiper 本体、数据源 `MyDataSource`、还是 `SwiperItemComponent`（可能涉及 props/state 设计）。
- 需要处理与现有 `onChange`、`maintainVisibleContentPosition`、数据增删的交互，避免重复加载、越界、或引入状态错乱。

## Case 2: `@performance/hp-arkui-use-reusable-component`（复杂 UI 抽取为可复用组件）

**规则意图（直观理解）**：把复杂/重复的 UI 片段抽取为 reusable component（或等价机制），让框架可以复用视图、降低重建成本。

**代表性代码片段**（列表/瀑布流 item 片段内嵌在页面 build 中）

```ts
build() {
  Column() {
    WaterFlow({ scroller:this.scroller }) {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {

          Text('N' + item)
            .id('show_flow_' + item)
            .width('100%')
            .height('34%')
            .fontSize(16)
            .textAlign(TextAlign.Center)
            .borderRadius(10)
        
        }
        .width('100%')
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    // ...
  }
}
```

**为什么很难用固定规则“安全自动修”**（需要 LLM 的点）
- 抽取组件时需要决定“组件边界”：哪些样式/状态（如 `colors`、`id`、尺寸）作为参数传入，哪些留在父组件。
- 需要保持语义与可测试性：`id('show_flow_'+item)` 这类测试/可观测性标识要如何在抽取后保留。
- 需要处理框架特性：是否需要 `@Reusable`、是否需要 `aboutToReuse`/复用参数更新、是否需要 `reuseId`（取决于组件是否有内部状态与复用策略）。

## Case 3: `@performance/avoid-overusing-custom-component-check`（把“轻量组件”改写为 `@Builder`）

**规则意图（直观理解）**：对于“很轻量、主要是样式/回调”的 UI 片段，优先用 `@Builder`，避免滥用 `@Component` 带来的额外结构成本。

**代表性代码片段**（一个包含 stateStyles、点击回调、样式方法的自定义组件）

```ts
@Preview
@Component
export struct MyComponent {
  private title: string = 'test'
  private func: () => void = () => {
  }

  @Styles
  pressedStyle() {
    .backgroundColor(0x238E23)
  }

  @Styles
  normalStyles() {
    .backgroundColor(0x0000FF)
  }

  build() {
    Text(this.title)
      .fontSize(10)
      .backgroundColor(0x0000FF)
      .fontColor(0xFFFFFF)
      .padding(5)
      .onClick(this.func)
      .stateStyles({
        pressed: this.pressedStyle(),
        normal: this.normalStyles()
      })
  }
}
```

**为什么很难用固定规则“安全自动修”**（需要 LLM 的点）
- 把 `@Component struct` 改成 `@Builder` 往往需要“状态提升/接口重设计”：`title`、`func`、`@Styles` 方法如何变成参数或闭包、在哪个文件/作用域定义。
- 需要同步修改所有调用点（不仅是本文件），并确保 `@Preview` / 导出符号/可访问性仍然正确。
- 需要保证行为一致（尤其是交互/样式状态机）：`stateStyles` 的绑定是否仍然可用、是否需要改写为等价写法。

## 写进 TOSEM 版论文的落点（A：定性为主）

- 在方法/动机章节加一个短小小节：“Why LLM is needed beyond deterministic templates”，每个 case 1 个片段 + 3–4 句解释（上面的“为什么需要 LLM 的点”可直接精简复用）。
- 强调工程闭环：LLM 负责产生候选重构，随后用编译、静态检查与（若有）测试进行筛选/回滚，降低生成式重构风险。
