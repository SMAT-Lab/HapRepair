import { WorkflowCycleEngine } from './no_cycle_workflow_bad_flow';

export class WorkflowCycleReporter {
  captureStage(stage: string, engine?: WorkflowCycleEngine): void {
    if (engine) {
      stageQueue.push(stage);
      engine.collectMetrics();
    }
  }
}

const stageQueue: Array<string> = [];
