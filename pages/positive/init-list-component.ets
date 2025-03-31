struct Greeting {
  @Builder myBuilder() {
    List().width(10).height(10);
  }
  build() {
    List().width(10).height(10);
  }
}

@Builder function myBuilder() {
  List().width(10).height(10)
}