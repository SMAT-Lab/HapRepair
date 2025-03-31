import { http } from '@kit.NetworkKit';
import { BusinessError } from '@ohos.base';
import taskpool from '@ohos.taskpool';

@Concurrent
function processRespTask(err: BusinessError, data: http.HttpResponse) {
  if (!err) {
    console.info('Result:' + data.result);
    console.info('code:' + data.responseCode);
    console.info('type:' + JSON.stringify(data.resultType));
    console.info('header:' + JSON.stringify(data.header));
    console.info('cookies:' + data.cookies);
  } else {
    console.info('error:' + JSON.stringify(err));
  }
}

let httpRequest = http.createHttp();
httpRequest.request("EXAMPLE_URL", async (err: Error, data: http.HttpResponse) => {
  let task = new taskpool.Task(processRespTask, data);
  await taskpool.execute(task);
});