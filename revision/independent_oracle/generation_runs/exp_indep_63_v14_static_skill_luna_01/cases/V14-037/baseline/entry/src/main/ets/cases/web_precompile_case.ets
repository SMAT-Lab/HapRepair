import webview from '@ohos.web.webview';

@Entry
@Component
struct DocumentationWebPage {
  private controller: webview.WebviewController = new webview.WebviewController();

  build() {
    Web({ src: 'https://example.org/docs.html', controller: this.controller })
  }
}
