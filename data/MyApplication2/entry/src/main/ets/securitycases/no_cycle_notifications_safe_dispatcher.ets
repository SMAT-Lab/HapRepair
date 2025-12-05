import { NotificationTracker } from './no_cycle_notifications_safe_tracker';

export class NotificationDispatcher {
  private tracker: NotificationTracker = new NotificationTracker();

  dispatch(event: string): void {
    this.tracker.track(event);
  }
}

const notifier: NotificationDispatcher = new NotificationDispatcher();
notifier.dispatch('alert');
