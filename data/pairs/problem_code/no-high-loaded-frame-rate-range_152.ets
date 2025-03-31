@Entry
@Component
struct AnimationComponent {
  setup() {
    animate({
      duration: 5000,
      frameRate: {
        expected: 144,
        min: 144,
        max: 144
      }
    });
  }
}