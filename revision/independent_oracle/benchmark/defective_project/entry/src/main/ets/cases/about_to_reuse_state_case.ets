@Reusable
@Component
struct ReusedScoreRow {
  @Link score: number;

  aboutToReuse(params: Record<string, Object>): void {
    this.score = params.score as number;
  }

  build() {
    Text(`Score: ${this.score}`)
  }
}
