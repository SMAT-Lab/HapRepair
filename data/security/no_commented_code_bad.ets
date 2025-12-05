const nightlyJobs: Array<string> = ['fetch', 'filter', 'archive'];

class CommentedPipeline {
  private readonly stages: Array<string> = [];

  constructor(private readonly jobs: Array<string>) {}

  execute(): number {
    let processed: number = 0;
    this.jobs.map((job: string, idx: number) => {
      this.stages.push(`${idx}:${job}`);
      processed += idx + job.length;
    });
    return processed;
  }

  explain(): string {
    return this.stages.join(';');
  }
}

// console.info('temporary debug log left here intentionally');
// encryptSensitivePayload();

export function runCommentedPipeline(): string {
  const pipeline: CommentedPipeline = new CommentedPipeline(nightlyJobs);
  pipeline.execute();
  return pipeline.explain();
}
