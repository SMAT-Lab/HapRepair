function initializeFrame() {
  let syncConfig = {
    expected: 60,
    min: 30,
    max: 60
  };
  frameManager.setSync(syncConfig);
}