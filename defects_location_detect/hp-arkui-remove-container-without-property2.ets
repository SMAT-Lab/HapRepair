interface User {
  name: string;
  age: number;
  city: string;
}

@Entry
@Component
struct UserTableComponent {
  @State users: Array<User> = [
    { name: 'Alice', age: 25, city: 'New York' },
    { name: 'Bob', age: 30, city: 'Los Angeles' },
    { name: 'Charlie', age: 35, city: 'Chicago' },
  ];

  build() {
    Column() {
      ForEach(this.users, (user: User) => {
        Flex() {
          Flex() {
            Text(user.name)
              .fontSize(16)
              .textAlign(TextAlign.Center)
              .padding({ top: 5, bottom: 5 })
          }
        }
      }, (user: User) => user.name)
    }
  }
}
