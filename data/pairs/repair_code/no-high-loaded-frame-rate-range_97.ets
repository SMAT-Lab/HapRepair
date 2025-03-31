@Widget
class FrameWidget {
  configure() {
    setRates({
      expected: 60,
      min: 30,
      max: 60
    });
  }
}