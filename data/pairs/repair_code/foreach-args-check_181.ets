@Entry
@Component
struct App {
  build() {
    List() {
      ForEach(this.members, (member, idx) => {
        MemberItem(member);
      }, (member) => member.email);
    }
    .width(10)
    .height(10)
  }
}