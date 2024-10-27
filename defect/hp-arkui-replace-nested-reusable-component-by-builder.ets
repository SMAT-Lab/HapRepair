@Entry
@Component
struct DashboardComponent {
  @State userStats: number[] = Array.from(Array(10), (_, i) => i + 1);

  build() {
    Column() {
      List() {
        LazyForEach(this.userStats, (stat: number) => {
          ListItem() {
            // 使用自定义的 UserInfo 组件嵌套渲染
            UserInfo({ stat: stat })
          }
        }, (stat: number) => stat.toString())
      }
      .width('100%')
      .height('100%')
    }
  }
}

@Reusable
@Component
struct UserInfo {
  @State stat: number = 0;

  aboutToReuse(params: ESObject): void {
    this.stat = params.stat as number;
  }

  build() {
    Column() {
      UserDetails({ stat: this.stat })
    }
    .width('100%')
    .height(100)
    .padding(10)
  }
}

@Component
struct UserDetails {
  @State stat: number = 0;

  aboutToReuse(params: ESObject): void {
    this.stat = params.stat as number;
  }

  build() {
    Row() {
      // 嵌套第三个自定义组件 UserStatsDisplay
      UserStatsDisplay({ stat: this.stat })
    }
    .padding(5)
  }
}

@Component
struct UserStatsDisplay {
  @State stat: number = 0;

  aboutToReuse(params: ESObject): void {
    this.stat = params.stat as number;
  }

  build() {
    Column() {
      Text('用户统计: ' + this.stat)
        .fontSize(20)
        .fontWeight(20)
    }
    .width('100%')
    .height(50)
  }
}