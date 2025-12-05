export class WorkflowPipelineReporter {
  private history: Array<string> = [];

  captureStage(stage: string): void {
    this.history.push(stage);
  }
}
