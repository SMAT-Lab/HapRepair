@Entry
@Component
struct App {
  build() {
    Grid() {
      ForEach(this.tasks, (task, index) => {
        TaskItem(task);
      });
    }
  }
}