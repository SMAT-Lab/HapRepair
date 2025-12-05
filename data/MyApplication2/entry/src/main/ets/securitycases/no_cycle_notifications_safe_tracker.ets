export class NotificationTracker {
  track(event: string): void {
    safeHistory.push(event);
  }
}

const safeHistory: Array<string> = [];
