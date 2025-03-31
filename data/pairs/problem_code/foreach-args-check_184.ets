@Entry
@Component
struct App {
  build() {
    HStack() {
      ForEach(this.events, (event, idx) => {
        EventItem(event);
      });
    }
  }
}