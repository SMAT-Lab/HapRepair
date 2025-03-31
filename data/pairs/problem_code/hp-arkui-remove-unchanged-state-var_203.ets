class Config {
  apiEndpoint: string = "https://api.example.com";
  timeout: number = 5000;
}

@Entry
@Component
struct AppConfig {
  @State config: Config = new Config();
  @State appTitle: string = "My Application";

  build() {
    Column() {
      Header(this.appTitle)
      // 配置详情显示
      ConfigView(this.config)
    }
    .padding(20)
  }
}
