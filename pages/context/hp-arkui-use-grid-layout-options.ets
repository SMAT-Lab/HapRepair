  layoutOptions: GridLayoutOptions = {
    regularSize: [1, 1],
    irregularIndexes: this.irregularData,
  };

  build() {
    Column({ space: 5 }) {
      Text('Use columnStart and columnEnd to set the GridItem size').fontColor(0xCCCCCC).fontSize(9).width('90%')
      Grid(this.scroller) {