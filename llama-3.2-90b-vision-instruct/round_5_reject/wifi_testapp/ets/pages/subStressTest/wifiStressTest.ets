
/**
 * wifi StressTest Page Of Wifi test
 */

@Entry
@Component
struct WifiStressTest {
  @State showList: boolean = false;
  @State showListItems: boolean = false;
  @State showListItem: boolean = false;
  private testItem: TestData = (router.getParams() as myParams).testItem
  @State deviceId: string = '';
  currentClick: number = - 1;
  @StorageLink('pairedMac') pairedMac: string = "";
  @StorageLink('stressNumber') stressNumber: number = 0;
  @StorageLink('stressTime') stressTime: number = 0;

  aboutToAppear() {
    AppStorage.setOrCreate('stressNumber' , this.stressNumber);
    AppStorage.setOrCreate('stressTime' , this.stressTime);
  }

  init() {
    console.log("init-----------")
  }

  build() {
    Column() {
      Stack({ alignContent : Alignment.TopStart }) {
        TestImageDisplay({ testItem : this.testItem })
        PageTitle({ testItem : this.testItem })
      }

      Stack().height("0.5vp").backgroundColor("#000000");
      Column() {
        Row() {
          Text("目标设备IP：")
            .fontSize("20vp")
            .height(40)
            .padding({ top : 5 })
            .width(ConfigData.WH_28_100)
            .align(Alignment.Start)

          TextInput({ text : this.pairedMac , placeholder : "请输入即将连接的设备MAC" })
            .height(40)
            .borderStyle(BorderStyle.Dashed)
            .backgroundColor($r("app.color.white"))
            .onChange((str) => {
              this.deviceId = str;
            })
            .width(ConfigData.WH_45_100)
            .padding({ top : 5 , left : 5 })
          Image($r('app.media.Switch'))
            .height(50)
            .padding({ top : 5 , bottom : 5 , left : 10 })
            .width(ConfigData.WH_15_100)
            .onClick(() => {
              this.showList = !this.showList;
            })
        }
      }
      .align(Alignment.Start)

      if ( this.showList ) {
        WifiDeviceSetting()
      } else {
        Column() {
          Scroll() {
            Column() {
              Text("stress测试：")
                .fontSize("17vp")
                .margin({ top : "5vp" , left : "15vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_100_100)
              Text("stress测试次数:" + this.stressNumber)
                .fontSize("17vp")
                .margin({ top : "5vp" , left : "15vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Text("stress测试时间：" + this.stressTime + "ms")
                .fontSize("17vp")
                .margin({ top : "5vp" , left : "15vp" })
                .textAlign(TextAlign.Start)
                .width(ConfigData.WH_90_100)
              Flex({ alignItems : ItemAlign.Center , justifyContent : FlexAlign.SpaceBetween }) {
                Button({ type : ButtonType.Normal , stateEffect : true }) {
                  Text('传输数据').fontSize("20vp").fontColor(0xffffff).margin({ left : 2 , right : 2 })
                }
                .borderRadius(8)
                .backgroundColor($r("app.color.blue"))
                .width(100)
                .height(60)
                .margin({ top : "5vp" , left : "10vp" , bottom : "5vp" })
                .align(Alignment.Start)
                .onClick(async(event: ClickEvent) => {
                  this.showListItems = true
                  this.showListItem = false
                })

                Button({ type : ButtonType.Normal , stateEffect : true }) {
                  Row() {
                    Text('上传下载').fontSize("20vp").fontColor(0xffffff).margin({ left : 2 , right : 2 })
                  }.alignItems(VerticalAlign.Center)
                }
                .borderRadius(8)
                .backgroundColor($r("app.color.blue"))
                .width(100)
                .height(60)
                .margin({ top : "5vp" , left : "10vp" , right : "10vp" , bottom : "5vp" })
                .align(Alignment.Center)
                .onClick(async(event: ClickEvent) => {
                  this.showListItem = true
                  this.showListItems = false
                })

                Button({ type : ButtonType.Normal , stateEffect : true }) {
                  Row() {
                    Text('返回接口').fontSize("20vp").fontColor(0xffffff).margin({ left : 2 , right : 2 })
                  }.alignItems(VerticalAlign.Bottom)
                }
                .borderRadius(8)
                .backgroundColor($r("app.color.blue"))
                .width(100)
                .height(60)
                .margin({ top : "5vp" , right : "10vp" , bottom : "5vp" })
                .align(Alignment.End)
                .onClick(async(event: ClickEvent) => {
                  this.showListItem = false
                  this.showListItems = false
                })
              }
              .backgroundColor($r("sys.color.ohos_id_color_sub_background"))
              .width(ConfigData.WH_100_100)
            }
            .width(ConfigData.WH_100_100)
          }
          .scrollBarWidth(10)
          .scrollBar(BarState.Auto)
        }
        .height(130)
        .width(ConfigData.WH_95_100)
        .backgroundColor($r("sys.color.ohos_id_color_foreground_contrary"))

        if ( this.showListItems == true && this.showListItem == false ) {
          socketTest()
        } else if ( this.showListItems == false && this.showListItem == true ) {
          upDownLoad()
        } else {
          StressContentTable({
            testItem : this.testItem , apiItems : initWifiStressData()
          })
        }
      }
    }
    .alignItems(HorizontalAlign.Center)
    .backgroundColor($r("app.color.lead"))
    .height(ConfigData.WH_100_100)
  }
}


@Component
struct upDownLoad {
  totalSize: number = 0 
  receivedSize: number = 0
  message: string = ''
  @State files: Array<string> = []
  @State uploads: Array<string> = []
  private downloadConfig: DownloadConfig = {
    url : 'http://192.168.62.127:8000' ,
    filePath : ""
  }

  aboutToAppear() {
    console.log(TAG , `enter Index aboutToAppear`)
    let context: Context = getContext(this) as common.UIAbilityContext;
    let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
    atManager.requestPermissionsFromUser(context , [ 'ohos.permission.INTERNET' ])
      .then((data: PermissionRequestResult) => {
        console.log(TAG , `grantPermission,requestPermissionsFromUser,result.requestCode=${ data }`)
      })
    this.files.push("http://192.168.62.127:8000")
    let httpRequest = http.createHttp()
    httpRequest.request(
      'http://192.168.0.123/?tpl=list&folders-filter=&recursive' ,
      {
        method : http.RequestMethod.GET ,
        header : {
          'Content-Type' : 'text/plain'
        }
      } , (err , data) => {
      console.log(TAG , `error = ${ JSON.stringify(err) }`)
      if ( !err ) {
        console.log(TAG , `data = ${ JSON.stringify(data) }`)
        let result: string = data.result.toString()
        console.log(TAG , `Result = ${ result }`)

        let tempFiles = result.split('\r\n')
        for ( let i = 0 ; i < tempFiles.length ; i ++ ) {
          let splitFiles = tempFiles[ i ].split('//')[ 1 ].split('/')
          console.log(TAG , `splitFiles = ${ JSON.stringify(splitFiles) }`)
          if ( splitFiles.indexOf('upload') === - 1 ) {
            this.files.push(tempFiles[ i ])
          }
        }
        console.log(TAG , `files = ${ JSON.stringify(this.files) }`)
      } else {
        console.log(TAG , `error: ${ JSON.stringify(err) }`)
        httpRequest.destroy()
      }
    })
  }

  build() {
    Column() {
      Scroll() {
        Column() {
          Text($r('app.string.choice_download_file'))
            .fontSize(25)
            .alignSelf(ItemAlign.Start)
            .margin({ top : 20 , left : 10 })
          ForEach(this.files , (item: string , index) => {
            Divider()
              .margin({ top : 20 })
            Row() {
              Text(item)
                .fontSize(25)
                .constraintSize({ maxWidth : ConfigData.WH_75_100 })
                .fontWeight(FontWeight.Bold)
                .margin({ top : 20 , left : 10 })
              Blank()

              Button($r('app.string.click_download'))
                .margin({ top : 20 , right : 10 })
                .onClick(async() => {
                  let fileindex = 1;
                  setInterval(() => {
                    this.downloadConfig.url = item
                    console.log(TAG , `downloadConfig.url = ${ JSON.stringify(this.downloadConfig.url) }`)
                    this.downloadConfig.filePath = `/data/storage/el2/base/haps/entry/cache/ ${
                    item.split('//')[ 1 ].split('/')[ 1 ] + fileindex }`
                    fileindex += 1
                    console.log(TAG , `downloadConfig.filePath = ${ JSON.stringify(this.downloadConfig.filePath) }`)
                    request.downloadFile(getContext() , this.downloadConfig , (err , downloadTask) => {
                      console.log(TAG , `download enter`)
                      if ( err ) {
                        console.log(TAG , `download err = ${ JSON.stringify(err) }`)
                        return
                      }
                      console.log(TAG , `download data = ${ JSON.stringify(downloadTask) }`)

                      console.log(TAG , `download end`)
                    })
                  } , 1000)

                })
            }
            .width(ConfigData.WH_100_100)

            if (index === this.files.length - 1) {
              Divider().margin({ top: 20 })
            }
          } , (item: string) => item)
          Button($r('app.string.upload_page'))
            .fontSize(30)
            .width(180)
            .height(50)
            .margin({ top : 30 })
            .key('uploadBtn')
            .onClick(() => {
              router.pushUrl({
                url : 'pages/Upload' ,
                params : { uploads : this.uploads }
              })
            })

        }
        .width(ConfigData.WH_100_100)
      }
      .constraintSize({ maxHeight : ConfigData.WH_87_100 })

      if ( this.receivedSize !== 0 && this.totalSize !== 0 && this.files.length > 0 ) {
        Row() {
          Text($r('app.string.download_progress'))
            .fontSize(25)
            .fontWeight(FontWeight.Bold)
            .margin({ left : 10 })
          Progress({ value : this.receivedSize , total : totalSize , type : ProgressType.Capsule }) 
            .color(Color.Grey)
            .width(ConfigData.WH_40_100)
            .height(10)
            .margin({ left : 10 })
          Text(`${ JSON.stringify(Math.floor(this.receivedSize / totalSize * 100)) }%`) 
            .margin({ left : 10 })
        }
        .margin({ top : 10 })
        .alignSelf(ItemAlign.Center)
      }
    }
  }
}