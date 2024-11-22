@Reusable
@Component
struct UserProfileComponent {
  @State name: string = '';
  @State age: number = 0;
  @Link email: string;

  aboutToReuse(params: Record<string, Object>): void {
    this.name = params.name as string;
    this.age = params.age as number;
    this.email = params.email as string;
  }

  build() {
    Column() {
      Text('姓名: ' + this.name)
        .fontSize(24)
      Text('年龄: ' + this.age)
        .fontSize(24)
      Text('邮箱: ' + this.email)
        .fontSize(24)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  name: string;
  age: number;
  email: string;
}

class UserData implements IDataSource {
  public users: GeneratedTypeLiteralInterface_1[] = [];

  setData(users: GeneratedTypeLiteralInterface_1[]) {
    this.users = users
  }

  totalCount(): number {
    throw new Error('Method not implemented.');
  }

  getData(index: number): GeneratedTypeLiteralInterface_1[]  {
    throw new Error('Method not implemented.');
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    throw new Error('Method not implemented.');
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    throw new Error('Method not implemented.');
  }

}

@Entry
@Component
struct UserListComponent {
  private users: UserData = new UserData()

  aboutToAppear(): void {
    // 模拟用户数据
    this.users.setData([
      { name: '张三', age: 30, email: 'zhangsan@example.com' },
      { name: '李四', age: 25, email: 'lisi@example.com' },
    ])
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.users, (user: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            UserProfileComponent({ name: user.name, age: user.age, email: user.email });
          }
          .width('100%')
          .height(100)
        }, (user: GeneratedTypeLiteralInterface_1) => user.email)
      }
      .cachedCount(2)
      .width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
  }
}
