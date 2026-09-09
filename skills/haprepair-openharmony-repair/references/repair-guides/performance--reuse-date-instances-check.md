# @performance/reuse-date-instances-check

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_a7ad4bb4e6e98960`

Avoid repeated creation of **Date** objects in loops or frequently called methods. Prioritize reusing existing instances or using timestamps for calculations.

### Triggering pattern

```arkts
aboutToAppear(): void {
  // Record the start time.
  this.startTime = Date.now();

  // Problem: Multiple Date objects are created every second.
  this.intervalId = setInterval(() => {

    // Create a Date object to display the current time.
    const now = new Date();
    this.currentTimeDisplay = now.toLocaleTimeString();

    // Create another Date object to calculate the elapsed time.
    const currentDate = new Date();
    const elapsedMs = currentDate.getTime() - this.startTime;

    // Create another Date object to format the elapsed time.
    const elapsedDate = new Date(elapsedMs);
    const hours = elapsedDate.getUTCHours();
    const minutes = elapsedDate.getUTCMinutes();
    const seconds = elapsedDate.getUTCSeconds();

    const elapsedFormatted = `${hours.toString().padStart(2, '0')}:${
    minutes.toString().padStart(2, '0')}:${
    seconds.toString().padStart(2, '0')}`;

    // Add to history.
    if (seconds % 10 === 0) {
      // Log every 10 seconds.
      this.elapsedTimes.push(`${elapsedFormatted}-${new Date().toLocaleString()}`);
    }
  }, 1000);
}
```

### Repair pattern

```arkts
aboutToAppear(): void {
  // Record the start time.
  this.startTime = Date.now();

  // Use a single Date instance per interval.
  this.intervalId = setInterval(() => {
    // Reuse a Date object.
    const now = new Date();
    this.currentTimeDisplay = now.toLocaleTimeString();

    // Use timestamp arithmetic to avoid creating extra Date objects.
    const elapsedMs = now.getTime() - this.startTime;
    this.elapsedTimeDisplay = this.formatElapsedTime(elapsedMs);

    // Add to history.
    const seconds = now.getSeconds();
    if (seconds % 10 === 0) {
      // Log every 10 seconds.
      this.elapsedTimes.push(`${this.elapsedTimeDisplay}-${this.formatCurrentTime(now)}`);
    }
  }, 1000);
}
```
