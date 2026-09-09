import webview from '@ohos.web.webview';

@Entry
@Component
struct DocumentationWebPage {
  private controller: webview.WebviewController = new webview.WebviewController();

  build() {
    Web({ src: 'https://example.org/docs.html', controller: this.controller })
      .onControllerAttached(async () => {
        try {
          await this.controller.precompileJavaScript(
            'https://example.org/docs.js',
            new ArrayBuffer(0)
          );
        } catch (error) {
          console.error(`Failed to precompile docs.js: ${error}`);
        }
      })
  }
}
