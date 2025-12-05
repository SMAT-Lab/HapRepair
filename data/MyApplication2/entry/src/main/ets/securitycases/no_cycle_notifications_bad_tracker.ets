import { IncidentDispatcher } from './no_cycle_notifications_bad_dispatcher';

export class IncidentTracker {
  track(event: string, dispatcher?: IncidentDispatcher): void {
    history.push(event);
    if (dispatcher) {
      dispatcher.getQueueLength();
    }
  }
}

const history: Array<string> = [];
