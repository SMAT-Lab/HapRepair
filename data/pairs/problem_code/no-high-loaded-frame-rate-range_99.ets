function processFrame() {
  videoConfig = {
    fps: {
      min: 210,
      expected: 210,
      max: 210
    }
  };
  videoPlayer.configure(videoConfig);
}