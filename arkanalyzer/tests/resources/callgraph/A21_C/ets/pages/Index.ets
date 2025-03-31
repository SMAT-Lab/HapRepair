import router from '@ohos.router';
import promptAction from '@ohos.promptAction'
import HttpUtl from '../pages/http';
import Function from '../pages/configue';

@Entry
@Component
struct Index {
  @State message: string = '智能家居'
  private userName: String=''
  private password0: String=''
  @State loadingWidth: number = 0

  build() {
      Column({space:10}) {
        Image($r("app.media.img_2"))
          .width(80)
          .height(80)
          .borderRadius(50)
          .margin({
            top:"120vp"
          })
        Text("欢迎使用智能风扇app")
          .fontColor(Color.Black)
          .fontSize(20)
          .margin({
            top:"15vp"
          })
        TextInput({placeholder:'请输入用户名'})
          .margin({bottom:'10vp'})
          .onChange((value: string) => {
            this.userName = value
          })

        TextInput({placeholder:'请输入密码'})
          .type(InputType.Password)
          .onChange((value: string) => {
            this.password0 = value
          })

        Button("登录").width("80%").type(ButtonType.Capsule).onClick(() => {
          router.pushUrl({url:'pages/MainMode'})
        }) .onClick(() => {
          if(this.userName.trim() == '') {
            promptAction.showToast({
              message: '账户不能为空！',
              duration: 2000
            })
            return
          }

          if(this.password.trim() == '') {
            promptAction.showToast({
              message: '密码不能为空！',
              duration: 2000
            })
            return
          }
          this.loadingWidth = 60
          setTimeout(() => {
            if(this.userName.trim() === 'admin' && this.password0.trim() === '123456'){
              router.replaceUrl({
                url: "pages/MainMode"
              })
            }else {
              promptAction.showToast({
                message: '账户或密码错误',
                duration: 2000
              })
            }
            this.loadingWidth = 0
          },2000)
        })
        Text("注册账号").fontColor("#007DFF").fontSize(14)
        LoadingProgress()
          .color('#007dfe')
          .height(this.loadingWidth)
          .width(this.loadingWidth)
        Blank()
    }
      .backgroundImage($r("app.media.bk3"))
      .backgroundImageSize({
        width:'100%',
        height:'100%'
      })
      .padding({
        left:'10vp',
        right:'10vp',
        bottom:'10vp'
      })
      .border({width:1})
    .width("100%")
    .height("100%")
  }
  login() {

    let url = "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens";

    let hw_user = {
      "auth": {
        "identity": {
          "methods": [
            "password"
          ],
          "password": {
            "user": {
              "name": "15503433788",
              "password": "yzh1593576842",
              "domain": {
                "name": "15503433788"
              }
            }
          }
        },
        "scope": {
          "domain": {
            "name": "15503433788"
          }
        }
      }
    }

    console.log("url",url);
    console.log("hw_user",hw_user);
    HttpUtl.post(url, hw_user, (result) => {
      console.log('result:',result);
      console.log('token:',globalThis.token);
      router.pushUrl({url:"pages/Log1"})
      Function.onInit()
    });

  }

}