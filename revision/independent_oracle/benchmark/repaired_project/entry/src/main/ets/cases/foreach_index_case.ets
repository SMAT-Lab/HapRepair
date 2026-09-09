interface TaskSummary {
  id: string;
  title: string;
}

@Component
struct TaskSummaryList {
  private tasks: TaskSummary[] = [
    { id: 'task-a', title: 'Review' },
    { id: 'task-b', title: 'Publish' }
  ];

  build() {
    List() {
      ForEach(this.tasks, (task: TaskSummary) => {
        ListItem() {
          Text(task.title)
        }
      }, (task: TaskSummary) => task.id)
    }
    .width('100%')
    .height(220)
  }
}
