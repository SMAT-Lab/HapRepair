import lottie from '@ohos/lottie';
import { AnimationItem } from '@ohos/lottie';

@Entry
@Component
struct SyncPulseBadge {
  private drawingSurface: CanvasRenderingContext2D = new CanvasRenderingContext2D();
  private pulse?: AnimationItem;

  private launchPulse(): void {
    this.pulse = lottie.loadAnimation({
      container: this.drawingSurface,
      renderer: 'canvas',
      autoplay: true,
      loop: false,
      name: 'sync-pulse',
      path: 'media/sync-pulse.json'
    });
  }

  onPageHide(): void {
    this.pulse?.destroy();
    this.pulse = undefined;
  }

  build() {
    Stack() {
      Canvas(this.drawingSurface)
        .width(72)
        .height(72)
        .onReady(() => this.launchPulse())
      Text('Syncing').fontSize(11)
    }
  }
}
