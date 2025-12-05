import { WorkflowPipelineReporter } from './no_cycle_workflow_safe_report';

export class WorkflowPipelineEngine {
  private reporter: WorkflowPipelineReporter = new WorkflowPipelineReporter();

  execute(stage: string): void {
    this.reporter.captureStage(stage);
  }
}

export const safePipeline: WorkflowPipelineEngine = new WorkflowPipelineEngine();
safePipeline.execute('validate');
