@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.userList, (user: User) => {
        ListItem() {
          UserComponent({ name: user.name, age: user.age });
        }
      }, (user: User) => JSON.stringify(user));
    }
  }
}