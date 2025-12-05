class ReportAssembler {
  constructor(private readonly segments: Array<string>) {}

  assemble(): string {
    return this.segments.reduce((acc: string, next: string) => `${acc}-${next}`, 'root');
  }
}

export function buildRiskReport(): string {
  const assembler: ReportAssembler = new ReportAssembler(['metrics', 'policy', 'result']);
  return assembler.assemble();
}
