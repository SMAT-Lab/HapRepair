function processFrame() {
  videoConfig = {
    fps: {
      min: 30,
      expected: 60,
      max: 60
    }
  };
  videoPlayer.configure(videoConfig);
}