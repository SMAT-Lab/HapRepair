import { http } from '@kit.NetworkKit';

@Entry
@Component
struct MyComponent {
  build() {
    Column() {
      Text('点击发送网络请求')
        .fontSize(20)
        .padding(10)

      Button('发送请求')
        .onClick(() => {
          let httpRequest = http.createHttp();
          httpRequest.request("EXAMPLE_URL", (err: Error, data: http.HttpResponse) => {
            if (!err) {
              console.info('Result: ' + data.result);
              console.info('Code: ' + data.responseCode);
            } else {
              console.info('Error: ' + JSON.stringify(err));
            }
          });
        })
        .padding(20)
        .backgroundColor(Color.Blue)
        .fontColor(Color.White)
    }
    .padding(20)
  }
}