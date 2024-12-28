
import ItemData from './ItemData';
export class MainViewModel {
  getSettingListData(): Array<ItemData> {
    let settingListData: ItemData[] = [
      new ItemData($r('app.string.setting_list_message'), $r('app.media.message'), $r('app.string.setting_toggle')),
      new ItemData($r('app.string.setting_list_menu'), $r('app.media.settings'), null),
      new ItemData($r('app.string.setting_list_about'), $r('app.media.aboutus'), null),
      new ItemData($r('app.string.setting_list_clear'), $r('app.media.clear'), null),
      new ItemData($r('app.string.setting_list_auther'), $r('app.media.privacy'), null),
    ]

    return settingListData;
  }
}

export default new MainViewModel();