
  @State color: string | number | Resource = "";

  aboutToReuse(params: Record<string, Object>): void {
    this.color = params.color as number;
  }

  build() {
    Column() {
      Text('This is the title')
      Text('This is the internal text')
        .fontColor(this.color)