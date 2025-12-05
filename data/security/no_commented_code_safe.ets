const buildSteps: Array<string> = ['auth', 'sync', 'persist'];

class CommentFreePipeline {
  private readonly history: Array<string> = [];

  constructor(private readonly steps: Array<string>) {}

  runOnce(): string {
    let result: string = '';
    this.steps.forEach((step: string, index: number) => {
      const record: string = `${index}-${step}`;
      this.history.push(record);
      result += record + '|';
    });
    return result;
  }

  summarize(): string {
    return this.history.join(',');
  }
}

export function buildDeploymentPlan(): string {
  const pipeline: CommentFreePipeline = new CommentFreePipeline(buildSteps);
  const lastRun: string = pipeline.runOnce();
  return `${lastRun}=>${pipeline.summarize()}`;
}
