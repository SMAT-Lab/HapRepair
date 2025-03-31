@Entry
@Component
struct AnimationComponent {
  setup() {
    animate({
      duration: 5000,
      frameRate: {
        expected: 60,
        min: 30,
        max: 60
      }
    });
  }
}