struct Greeting {
  @Builder myBuilder() {
    List().width(10)
  }
  build() {
    List().height(10);
  }
}

@Builder function myBuilder() {
  List().width(10)
}