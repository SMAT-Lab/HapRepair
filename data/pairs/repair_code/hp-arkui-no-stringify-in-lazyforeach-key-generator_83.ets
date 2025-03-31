@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.taskList, (task: Task) => {
        ListItem() {
          TaskComponent({ title: task.title });
        }
      }, (task: Task) => task.taskId.toString());
    }
  }
}