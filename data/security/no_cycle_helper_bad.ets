import { renderCircular } from './no_cycle_bad';

export class LoopingHelper {
  constructor(private readonly name: string) {}

  describe(): string {
    return `helper:${this.name}`;
  }

  triggerLoop(): string {
    return renderCircular();
  }
}

export function loopHelperName(input: string): string {
  const helper: LoopingHelper = new LoopingHelper(input);
  helper.triggerLoop();
  return helper.describe();
}
