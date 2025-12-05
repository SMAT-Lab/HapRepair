
import http from '@ohos.net.http';

export default  {
  get(url: string, fn: Function): void {
    const httpRequest = http.createHttp();
    httpRequest.request(url, this.setOptions(http.RequestMethod.GET, null), (err, data) => {
      if (!err) {
        const result = JSON.parse(JSON.stringify(data.result));
        fn(result);
      } else {
        console.info('error:' + JSON.stringify(err));
        httpRequest.destroy();
      }
    });
  },

  post(url: string, params: any, fn: Function): void {
    let httpRequest = http.createHttp();
    httpRequest.request(url, this.setOptions(http.RequestMethod.POST, params), (err, data) => {
      if (!err) {
        if (url.indexOf('/auth/tokens') > 0) {
          // Assuming getApp().globalData.token is available globally
          globalThis.token = data.header['x-subject-token'];
        }
        fn(JSON.parse(JSON.stringify(data.result)));
      } else {
        console.info('error:' + JSON.stringify(err));
        httpRequest.destroy();
      }
    });
  },

  setOptions(method: string, params: any): any {
    const options = {
      method: method === 'get' ? http.RequestMethod.GET : http.RequestMethod.POST,
      header: {
        'Content-Type': 'application/json',
        'X-Auth-Token': globalThis.token,
      },
      expectDataType: http.HttpDataType.STRING,
      extraData : params,
    };

    return options;
  }
}