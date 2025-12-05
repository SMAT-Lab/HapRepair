import { WorkflowCycleReporter } from './no_cycle_workflow_bad_report';

export class WorkflowCycleEngine {
  constructor(private reporter: WorkflowCycleReporter) {}

  execute(stage: string): void {
    this.reporter.captureStage(stage, this);
  }

  collectMetrics(): number {
    return stageCache.length;
  }
}

const stageCache: Array<string> = ['ingest'];
export const legacyEngine: WorkflowCycleEngine = new WorkflowCycleEngine(new WorkflowCycleReporter());
legacyEngine.execute('validate');
