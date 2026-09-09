# @performance/js-code-cache-by-precompile-check

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_e77007a2b85dde4e`

By pre-compiling the used JavaScript files into bytecode and caching it locally, you can significantly reduce the time required for both the first and second page loads.

### Triggering pattern

```arkts
import { webview } from '@kit.ArkWeb';
import { hiTraceMeter } from '@kit.PerformanceAnalysisKit';

@Entry
@Component
struct JsCodeCacheByPrecompileCheckReport {
  controller: webview.WebviewController = new webview.WebviewController();

  build() {
    Column() {
      Button('加载页面')
        .onClick(() => {
          hiTraceMeter.startTrace('unPrecompileJavaScript', 1);
          this.controller.loadUrl('https://www.example.com/b.html');
        })
      // warning line
      Web({ src: 'https://www.example.com/a.html', controller: this.controller })
        .fileAccess(true)
        .onPageBegin((event) => {
          console.log(`load page begin: ${event?.url}`);
        })
        .onPageEnd((event) => {
          hiTraceMeter.finishTrace('unPrecompileJavaScript', 1);
          console.log(`load page end: ${event?.url}`);
        })
    }
  }
}
```

### Repair pattern

```arkts
import { webview } from '@kit.ArkWeb';

interface Config {
  url: string,
  localPath: string,
  options: webview.CacheOptions
}

@Entry
@Component
struct JsCodeCacheByPrecompileCheckNoReport {
  controller: webview.WebviewController = new webview.WebviewController();
  configs: Array<Config> = [
    {
      url: 'https://www.example.com/example.js',
      localPath: 'example.js',
      options: {
        responseHeaders: [
          { headerKey: 'E-Tag', headerValue: 'xxx' },
          { headerKey: 'Last-Modified', headerValue: 'Web, 21 Mar 2024 10:38:41 GMT' }
        ]
      }
    }
  ]

  build() {
    Column() {
      Web({ src: 'https://www.example.com/a.html', controller: this.controller })
        .onControllerAttached(async () => {
          for (const config of this.configs) {
            let content = getContext().resourceManager.getRawFileContentSync(config.localPath);
            try {
              this.controller.precompileJavaScript(config.url, content, config.options)
                .then((errCode: number) => {
                  console.log('precompile successfully!' );
                }).catch((errCode: number) => {
                console.error('precompile failed.' + errCode);
              })
            } catch (err) {
              console.error('precompile failed!.' + err.code + err.message);
            }
          }
        })
    }
  }
}
```
