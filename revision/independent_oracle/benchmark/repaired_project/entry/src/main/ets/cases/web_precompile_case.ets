import webview from '@ohos.web.webview';

interface ScriptCacheConfig {
  url: string;
  localPath: string;
  options: webview.CacheOptions;
}

@Entry
@Component
struct DocumentationWebPage {
  controller: webview.WebviewController = new webview.WebviewController();
  configs: Array<ScriptCacheConfig> = [{
    url: 'https://example.org/app.js',
    localPath: 'app.js',
    options: {
      responseHeaders: [
        { headerKey: 'E-Tag', headerValue: 'revision-1' }
      ]
    }
  }];

  build() {
    Web({ src: 'https://example.org/docs.html', controller: this.controller })
      .onControllerAttached(async () => {
        for (const config of this.configs) {
          const content: Uint8Array = getContext().resourceManager.getRawFileContentSync(config.localPath);
          this.controller.precompileJavaScript(config.url, content, config.options)
            .then(() => console.info('Precompile complete'))
            .catch(() => console.error('Precompile failed'));
        }
      })
  }
}
