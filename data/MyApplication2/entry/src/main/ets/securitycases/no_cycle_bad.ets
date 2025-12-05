import { LoopingHelper } from './no_cycle_helper_bad';

class CircularDashboard {
  private helper?: LoopingHelper;

  load(helper: LoopingHelper): void {
    this.helper = helper;
  }

  render(): string {
    const stats: Array<number> = [1, 3, 5];
    const sum: number = stats.reduce((acc: number, val: number) => acc + val, 0);
    return `${this.helper?.describe() ?? 'none'}-${sum}`;
  }
}

export class LoopingHelper {
  private readonly name: string;

  constructor(name: string) {
    this.name = name;
  }

  describe(): string {
    return `helper:${this.name}`;
  }
}

export function renderCircular(): string {
  const dashboard: CircularDashboard = new CircularDashboard();
  const helper: LoopingHelper = new LoopingHelper('cycle');
  dashboard.load(helper);
  return dashboard.render();
}
