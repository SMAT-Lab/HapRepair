import { IncidentTracker } from './no_cycle_notifications_bad_tracker';

export class IncidentDispatcher {
  private tracker: IncidentTracker = new IncidentTracker();

  dispatch(event: string): void {
    this.tracker.track(event, this);
  }

  getQueueLength(): number {
    return events.length;
  }
}

const events: Array<string> = ['boot'];
export const dispatcher: IncidentDispatcher = new IncidentDispatcher();
dispatcher.dispatch('alert');
