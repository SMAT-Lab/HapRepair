
import HttpUtil from '../pages/http';
//import router from '@ohos.router';
import promptAction from '@ohos.promptAction';


export default {
  data: {
    // 以下修改成自己的配置
    endpoint: 'https://a66d1506bf.st1.iotda-app.cn-north-4.myhuaweicloud.com',  // 修改自己的网络地址
    project_id: '65a2421d2ccc1a5838740832',  // 在‘我的凭证’中找
    device_id: '65a2421d2ccc1a5838740832_123456789',
    service_id: 'Agriculture',
    light_command: 'led_control',
    motor_command: 'fan_control',
    // 设备属性
    temperature: 0,
    humidity: 0,
    luminance: 0,
    lightStatus: 'OFF',
    motorStatus: 'OFF',
    // 为了UI好看，不用管这个属性
    pm25: 0,
    airQuality: '--',
  },
  onInit() {
    setInterval(() => {
      this.initProperties();
    }, 3000);
  },
  initProperties() {
    // let url = '/v5/iot/' + project_id + '/devices/' + device_id + '/properties?service_id=' + service_id;
    const url = '/v5/iot/' + this.project_id + '/devices/' + this.device_id + '/shadow';
    HttpUtil.get(this.endpoint + url, (result) => {
      console.log('Properties:', JSON.stringify(result.shadow[0].reported.properties));
      this.temperature = result.shadow[0].reported.properties.Temperature;
      this.humidity = result.shadow[0].reported.properties.Humidity;
      this.luminance = result.shadow[0].reported.properties.Luminance;
      this.lightStatus = result.shadow[0].reported.properties.LightStatus;
      this.motorStatus = result.shadow[0].reported.properties.MotorStatus;
      // 为了UI好看，不用管这个属性
      this.pm25 = Math.floor(Math.random() * 100 + 1);
      this.setAirQuality(this.pm25);
    });
  },
  switchLight() {
    const url = '/v5/iot/' + this.project_id + '/devices/' + this.device_id + '/commands';
    const Light = this.lightStatus == 'OFF' ? 'ON' : 'OFF';
    const commands = {
      service_id: this.service_id,
      command_name: this.light_command,
      "paras": {
        Light
      }
    };
    HttpUtil.post(this.endpoint + url, commands, (result) => {
      if (result.response.result_code != 0) {
        promptAction.showToast({
          message: '操作失败'
        });
      } else {
        this.lightStatus = Light;
      }
    });
  },
  switchMotor() {
    const url = '/v5/iot/' + this.project_id + '/devices/' + this.device_id + '/commands';
    const Motor = this.motorStatus == 'OFF' ? 'ON' : 'OFF';
    const commands = {
      service_id: this.service_id,
      command_name: this.motor_command,
      "paras": {
        Motor
      }
    };
    HttpUtil.post(this.endpoint + url, commands, (result) => {
      // this.motorStatus = result.response.response.paras.Motor;
      if (result.response.result_code != 0) {
        promptAction.showToast({
          message: '操作失败'
        });
      } else {
        this.motorStatus = Motor;
      }
    });
  },
  setAirQuality(pm25) {
    console.log('pm25', pm25);
    if (pm25 > 0 && pm25 <= 40) {
      this.airQuality = '优';
    } else if (pm25 > 40 && pm25 <= 70) {
      this.airQuality = '良';
    } else {
      this.airQuality = '差';
    }
  },
};