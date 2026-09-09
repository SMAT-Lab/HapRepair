function storeLateReading(value: number): number[] {
  const readings: number[] = Array.from({ length: 4096 }, () => undefined as number);
  readings[4095] = value;
  return readings;
}
