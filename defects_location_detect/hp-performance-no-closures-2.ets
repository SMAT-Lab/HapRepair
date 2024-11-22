interface UserSettings {
  theme: string;
  notificationsEnabled: boolean;
}

let userSettings: UserSettings = {
  theme: 'dark',
  notificationsEnabled: true,
};

function applySettings() {
  // 直接访问全局变量 userSettings
  console.log(`Applying settings: Theme - ${userSettings.theme}, Notifications enabled: ${userSettings.notificationsEnabled}`);

  // 执行一些设置操作
  if (userSettings.theme === 'dark') {
    console.log('Applying dark theme...');
  }
  if (userSettings.notificationsEnabled) {
    console.log('Enabling notifications...');
  }
}

applySettings();
