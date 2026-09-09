# @performance/lottie-animation-destroy-check

Static repair references: 15. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0067180d95b841f2`

在动画完成后及时销毁以防止内存浪费。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct FinishAnimation {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationCtx)
      .width(160)
      .height(160)
      .onReady(() => {
        // 无动画销毁
        lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'finish_anim.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct FinishAnimation {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private currentAnimation?: AnimationItem;

  build() {
    Canvas(this.animationCtx)
      .width(160)
      .height(160)
      .onReady(() => {
        // 无动画销毁
        lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'finish_anim.json'
        });
      })

      .onDisAppear(() => {
        this.currentAnimation?.destroy();
        this.currentAnimation = undefined;
      })
  }

}
```

### Rationale

未对完成的动画进行销毁，这会影响性能和内存使用。

## Example 2: `pair_1241c9586d97e8b5`

当使用lottie加载动画时，一般需要先通过lottie.loadAnimation将动画加载到内存，动画执行完毕后需要在合适的时机（例如：onDisAppear，onPageHide，aboutToDisappear）通过调用animationItem的destroy方法将单个动画销毁或者调用lottie.destroy()方法将当前页面所有动画销毁，如果动画未被销毁就会造成资源浪费，影响应用性能体验。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample2 {
  private animationController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.animationController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.animationController,
          renderer: 'canvas',
          loop: false,
          autoplay: true,
          path: 'animation2.json'
        });
      });
  }

  onDisAppear() {
    // 未销毁任何动画
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample2Fixed {
  private animationController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.animationController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.animationController,
          renderer: 'canvas',
          loop: false,
          autoplay: true,
          path: 'animation2.json'
        });
      });
  }

  onDisAppear() {
    if (this.animationItem) {
      this.animationItem.destroy(); // 清除动画确保内存释放
      this.animationItem = null;
    }
  }
}
```

### Rationale

动画执行完毕后使用destroy方法销毁Animation

## Example 3: `pair_14623e4916a0130d`

当使用lottie加载动画时，一般需要先通过lottie.loadAnimation将动画加载到内存，动画执行完毕后需要在合适的时机（例如：onDisAppear，onPageHide，aboutToDisappear）通过调用animationItem的destroy方法将单个动画销毁或者调用lottie.destroy()方法将当前页面所有动画销毁，如果动画未被销毁就会造成资源浪费，影响应用性能体验。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

// 动画播放的起始帧
const FRAME_START: number = 60; 
// 动画播放的终止帧
const FRAME_END: number = 120; 

//调用了销毁，但是不是全部销毁，上报
@Entry
@Component
struct LottieAnimation5 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  // 动画名称 
  private politeChicky: string = 'politeChicky'; 
  // hap包内动画资源文件路径，仅支持json格式
  private politeChickyPath: string = 'media/politeChicky.json'; 
  private animateItem: AnimationItem | null = null;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .backgroundColor(Color.Gray)
      .borderRadius(3)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name1',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
      .onClick(()=> {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name2',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
      .onDisAppear(()=>{
        //上报lottie,只销毁一个
        lottie.destroy('anim_name2');
      })
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

// 动画播放的起始帧
const FRAME_START: number = 60; 
// 动画播放的终止帧
const FRAME_END: number = 120; 

@Entry
@Component
struct LottieAnimation2 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  // 动画名称
  private politeChicky: string = 'politeChicky'; 
  // hap包内动画资源文件路径，仅支持json格式
  private politeChickyPath: string = 'media/politeChicky.json'; 
  private animateItem: AnimationItem | null = null;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .borderRadius(3)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name1',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
      .onClick(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name2',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
  }

  onPageHide(): void {
    lottie.destroy();
  }
}
```

### Rationale

加载了多个name不一样的animation，lottie.destroy('anim_name2')只销毁了其中一个animation，仍然会造成资源的浪费, 建议使用lottie.destroy()来销毁所有animation

## Example 4: `pair_29b5c6870a6b6e3e`

在生命周期结束时确保动画销毁以优化性能。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct LifecycleAnimation {
  private animationSurface: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationSurface)
      .width(150)
      .height(150)
      .onReady(() => {
        // 没有销毁操作
        lottie.loadAnimation({
          container: this.animationSurface,
          renderer: 'html',
          autoplay: true,
          path: 'lifecycle.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct LifecycleAnimationFixed {
  private animationSurface: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private activeAnimation?: AnimationItem;

  build() {
    Canvas(this.animationSurface)
      .width(150)
      .height(150)
      .onReady(() => {
        this.activeAnimation = lottie.loadAnimation({
          container: this.animationSurface,
          renderer: 'html',
          autoplay: true,
          path: 'lifecycle.json'
        });
      });
  }

  onPageHide() {
    this.activeAnimation?.destroy();
    this.activeAnimation = undefined;
  }
}
```

### Rationale

缺乏必要的销毁操作，导致动画未释放占用的资源。

## Example 5: `pair_32c0fc76d795012a`

在动画执行完毕后需要销毁动画以防止内存泄漏。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct SingleAnimation {
  private context: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation: AnimationItem | undefined;

  build() {
    Canvas(this.context)
      .width(100)
      .height(100)
      .onReady(() => {
        // 无销毁逻辑
        this.animation = lottie.loadAnimation({
          container: this.context,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'simple_anim.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct SingleAnimationFixed {
  private context: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation: AnimationItem | undefined;

  build() {
    Canvas(this.context)
      .width(100)
      .height(100)
      .onReady(() => {
        this.animation = lottie.loadAnimation({
          container: this.context,
          renderer: 'svg',
          loop: false,
          autoplay: true,
          path: 'simple_anim.json'
        });
      });
  }

  aboutToDisappear() {
    this.animation?.destroy();
    this.animation = undefined;
  }
}
```

### Rationale

动画执行完后未进行销毁导致内存占用无法释放。

## Example 6: `pair_56a42e1f00edae31`

使用lottie加载动画时，应在适当时机销毁以避免资源浪费。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct ExampleComponent {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation?: AnimationItem;

  build() {
    Canvas(this.myController)
      .width(150)
      .height(150)
      .onReady(() => {
        // 未在页面消失时销毁动画
        this.animation = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          autoplay: true,
          path: 'example.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct ExampleComponentFixed {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animation?: AnimationItem;

  build() {
    Canvas(this.myController)
      .width(150)
      .height(150)
      .onReady(() => {
        this.animation = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          autoplay: true,
          path: 'example.json'
        });
      });
  }

  onDisAppear() {
    this.animation?.destroy();
    this.animation = null;
  }
}
```

### Rationale

未调用任何method来销毁动画，会造成内存泄漏。

## Example 7: `pair_5f218dbc06edf4ef`

当使用lottie加载动画时，一般需要先通过lottie.loadAnimation将动画加载到内存，动画执行完毕后需要在合适的时机（例如：onDisAppear，onPageHide，aboutToDisappear）通过调用animationItem的destroy方法将单个动画销毁或者调用lottie.destroy()方法将当前页面所有动画销毁，如果动画未被销毁就会造成资源浪费，影响应用性能体验。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

const FRAME_START: number = 60;
const FRAME_END: number = 120;

@Entry
@Component
struct LottieAnimation1 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private politeChicky: string = 'politeChicky';
  private politeChickyPath: string = 'media/politeChicky.json';
  private animateItem?: AnimationItem;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .backgroundColor(Color.Gray)
      .borderRadius(3)
      .onReady(() => {
        //告警
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: this.politeChicky,
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

const FRAME_START: number = 60;
const FRAME_END: number = 120;

@Entry
@Component
struct LottieAnimation1 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private politeChicky: string = 'politeChicky';
  private politeChickyPath: string = 'media/politeChicky.json';
  private animateItem?: AnimationItem;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .borderRadius(3)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: this.politeChicky,
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
      .onDisAppear(() => {
        this.animateItem?.destroy();//只加载了一个Animation，可以使用animateItem的destroy接口
      })
  }
}
```

### Rationale

动画执行完毕后使用destroy方法销毁Animation

## Example 8: `pair_6daeac7609c01f93`

确保动画在页面生命周期结束时被销毁，以提高性能。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct MissedDestroy {
  private renderController: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.renderController)
      .width(180)
      .height(180)
      .onReady(() => {
        // 未销毁动画
        lottie.loadAnimation({
          container: this.renderController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'missed_destroy.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct MissedDestroyFixed {
  private renderController: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.renderController)
      .width(180)
      .height(180)
      .onReady(() => {
        lottie.loadAnimation({
          container: this.renderController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'missed_destroy.json'
        });
      });
  }

  onPageHide() {
    lottie.destroy();
  }
}
```

### Rationale

未针对动画进行销毁操作，在页面消失时会占用资源。

## Example 9: `pair_8f502d5a51bf6f79`

加载多个动画时，销毁未使用的动画资源。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct MultipleAnimationsComponent {
  private animationContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationContext)
      .width(200)
      .height(200)
      .onReady(() => {
        // 未销毁多余动画
        lottie.loadAnimation({
          container: this.animationContext,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'multi_anim.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

import { AnimationItem } from '@ohos/lottie';
@Entry
@Component
struct MultipleAnimationsComponentFixed {
  private animationContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private runningAnimations: AnimationItem[] = [];

  build() {
    Canvas(this.animationContext)
      .width(200)
      .height(200)
      .onReady(() => {
        const loadedAnim = lottie.loadAnimation({
          container: this.animationContext,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'multi_anim.json'
        });
        this.runningAnimations.push(loadedAnim);
      })
      .onDisAppear(() => {
        this.runningAnimations.forEach(anim => anim.destroy());
        this.runningAnimations = [];
      })
  }
}
```

### Rationale

多余动画未被销毁，造成资源浪费。

## Example 10: `pair_92ad2cb16a03c866`

应在组件卸载时销毁lottie动画以释放资源。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct AnimationComponentFixed {
  private renderContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animateItem?: AnimationItem;

  build() {
    Canvas(this.renderContext)
      .width(120)
      .height(120)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.renderContext,
          renderer: 'canvas',
          autoplay: true,
          path: 'anim.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct AnimationComponentFixed {
  private renderContext: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animateItem?: AnimationItem;

  build() {
    Canvas(this.renderContext)
      .width(120)
      .height(120)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.renderContext,
          renderer: 'canvas',
          autoplay: true,
          path: 'anim.json'
        });
      });
  }

  onDeactivate() {
    this.animateItem?.destroy();
    this.animateItem = undefined;
  }
}
```

### Rationale

没有在组件卸载时销毁动画，可能导致内存泄漏和资源消耗。

## Example 11: `pair_aa0b509b59eab8e0`

应在特定生命周期事件中销毁所有动画以节省资源。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct IncompleteDestroy {
  private drawingController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.drawingController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.drawingController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          path: 'incomplete.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct IncompleteDestroyFixed {
  private drawingController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.drawingController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.drawingController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          path: 'incomplete.json'
        });
      });
  }

  onPageHide() {
    if (this.animationItem) {
      this.animationItem.destroy();
      this.animationItem = null;
    }
  }
}
```

### Rationale

页面没有妥善销毁动画，可能导致内存浪费。

## Example 12: `pair_cf673440031e8bdb`

当使用lottie加载动画时，一般需要先通过lottie.loadAnimation将动画加载到内存，动画执行完毕后需要在合适的时机（例如：onDisAppear，onPageHide，aboutToDisappear）通过调用animationItem的destroy方法将单个动画销毁或者调用lottie.destroy()方法将当前页面所有动画销毁，如果动画未被销毁就会造成资源浪费，影响应用性能体验。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample1 {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.myController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      })
      .onClick(() => {
        // 每次点击都加载新动画但未销毁之前的动画
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      })
      .onDisAppear(() => {
        lottie.destroy('non_existing_anim'); // 錯誤的动画名称或者只销毁一个动画
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct LottieExample1Fixed {
  private myController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationItem: AnimationItem | null = null;

  build() {
    Canvas(this.myController)
      .width(200)
      .height(200)
      .onReady(() => {
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      })
      .onClick(() => {
        if (this.animationItem) {
          this.animationItem.destroy();  // 首先销毁现有动画实例
        }
        this.animationItem = lottie.loadAnimation({
          container: this.myController,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: 'animation.json'
        });
      });
  }

  onDisAppear() {
    lottie.destroy(); // 在页面消失时销毁所有相关动画
  }
}
```

### Rationale

页面有可能加载了多个name不一样的animation，lottie.destroy('non_existing_anim')只销毁了其中一个animation，仍然会造成资源的浪费, 建议使用lottie.destroy()来销毁所有animation

## Example 13: `pair_d4ea73c2e0933076`

动画在页面隐藏时需要适时销毁。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct PersistentAnimation {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();

  build() {
    Canvas(this.animationCtx)
      .width(100)
      .height(100)
      .onReady(() => {
        // 页面隐藏时未执行销毁
        lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'canvas',
          autoplay: true,
          path: 'persistent.json'
        });
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct PersistentAnimationFixed {
  private animationCtx: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animationInstance?: AnimationItem;

  build() {
    Canvas(this.animationCtx)
      .width(100)
      .height(100)
      .onReady(() => {
        this.animationInstance = lottie.loadAnimation({
          container: this.animationCtx,
          renderer: 'canvas',
          autoplay: true,
          path: 'persistent.json'
        });
      });
  }

  onHidePage() {
    this.animationInstance?.destroy();
    this.animationInstance = undefined;
  }
}
```

### Rationale

页面隐藏时动画未被销毁，导致可能的内存泄漏。

## Example 14: `pair_e1fe3c2672becf6b`

当使用lottie加载动画时，一般需要先通过lottie.loadAnimation将动画加载到内存，动画执行完毕后需要在合适的时机（例如：onDisAppear，onPageHide，aboutToDisappear）通过调用animationItem的destroy方法将单个动画销毁或者调用lottie.destroy()方法将当前页面所有动画销毁，如果动画未被销毁就会造成资源浪费，影响应用性能体验。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

// 动画播放的起始帧
const FRAME_START: number = 60; 
// 动画播放的终止帧
const FRAME_END: number = 120; 

//调用多次loadAnimation，但是只在onDisAppear销毁一次
@Entry
@Component
struct LottieAnimation4 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  // 动画名称  
  private politeChicky: string = 'politeChicky'; 
  // hap包内动画资源文件路径，仅支持json格式
  private politeChickyPath: string = 'media/politeChicky.json'; 
  private animateItem: AnimationItem | null = null;
  // 初始化点击次数 
  @State times: number = 0; 

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      // 动画
      Canvas(this.politeChickyController)
        .width(160)
        .height(160)
        .backgroundColor(Color.Gray)
        .borderRadius(3)
        .onReady(() => {
          this.animateItem = lottie.loadAnimation({
            container: this.politeChickyController,
            renderer: 'canvas',
            loop: true,
            autoplay: true,
            name: this.politeChicky,
            path: this.politeChickyPath,
            initialSegment: [FRAME_START, FRAME_END]
          })
        })
        .onClick(() => {
          this.animateItem = lottie.loadAnimation({
            container: this.politeChickyController,
            renderer: 'canvas',
            loop: true,
            autoplay: true,
            name: this.politeChicky,
            path: this.politeChickyPath,
            initialSegment: [FRAME_START, FRAME_END]
          })
          this.times++;
        })
        .onDisAppear(()=> {
          //上报此处animateItem，描述description不一样，如果无法找到动画名称，则直接建议用lottie.destory
          this.animateItem?.destroy();
        })
      // 响应动画的文本
      Text('text')
        .fontSize(16)
        .margin(10)
        .fontColor(Color.White)
    }.margin({ top: 20 })
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

// 动画播放的起始帧
const FRAME_START: number = 60; 
// 动画播放的终止帧
const FRAME_END: number = 120; 

@Entry
@Component
struct LottieAnimation2 {
  private politeChickyController: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  // 动画名称
  private politeChicky: string = 'politeChicky'; 
  // hap包内动画资源文件路径，仅支持json格式
  private politeChickyPath: string = 'media/politeChicky.json'; 
  private animateItem: AnimationItem | null = null;

  build() {
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .borderRadius(3)
      .onReady(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name1',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
      .onClick(() => {
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: 'anim_name2',
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })
  }

  onPageHide(): void {
    lottie.destroy();
  }
}
```

### Rationale

加载了多个name不一样的animation，不能直接使用this.animateItem?.destroy(), 建议使用lottie.destroy()来销毁

## Example 15: `pair_f108dd690d4e32a8`

确保所有加载的动画在不需要时被正确销毁。

### Triggering pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct MultipleAnimations {
  private controller: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animations: AnimationItem[] = [];

  build() {
    Canvas(this.controller)
      .width(180)
      .height(180)
      .onReady(() => {
        // 加载多个动画但未销毁
        for (let i = 0; i < 3; i++) {
          const anim = lottie.loadAnimation({
            container: this.controller,
            renderer: 'canvas',
            loop: true,
            autoplay: true,
            path: `animation_${i}.json`
          });
          this.animations.push(anim);
        }
      });
  }
}
```

### Repair pattern

```arkts
import lottie from '@ohos/lottie';

@Entry
@Component
struct MultipleAnimationsFixed {
  private controller: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private animations: AnimationItem[] = [];

  build() {
    Canvas(this.controller)
      .width(180)
      .height(180)
      .onReady(() => {
        for (let i = 0; i < 3; i++) {
          const anim = lottie.loadAnimation({
            container: this.controller,
            renderer: 'canvas',
            loop: true,
            autoplay: true,
            path: `animation_${i}.json`
          });
          this.animations.push(anim);
        }
      });
  }

  onPageHide() {
    this.animations.forEach(anim => anim.destroy());
    this.animations = [];
  }
}
```

### Rationale

未对加载的动画进行销毁操作，可能会导致资源泄漏。
