@Reusable
@Component
struct ReusedScoreRow {
  @State score: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.score = params.score as number;
  }

  build() {
    Text(`Score: ${this.score}`)
  }
}
