@Entry
@Component
struct App {
  build() {
    List() {
      ForEach(this.members, (member, idx) => {
        MemberItem(member);
      });
    }
  }
}