function storeLateReading(value: number): number[] {
  const readings: number[] = new Array(4096);
  readings[4095] = value;
  return readings;
}
