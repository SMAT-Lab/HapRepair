import { provisionWeakKey } from './specified_interface_call_chain_helper_bad';

class KeyChainScheduler {
  constructor(private readonly userIds: Array<string>) {}

  async triggerForAll(): Promise<void> {
    for (const id of this.userIds) {
      await this.dispatchJob(id);
    }
  }

  private async dispatchJob(userId: string): Promise<void> {
    const alias: string = `audit-${userId}`;
    await provisionWeakKey(alias);
  }
}

export async function auditProvisionFlow(): Promise<void> {
  const scheduler: KeyChainScheduler = new KeyChainScheduler(['alpha', 'beta']);
  await scheduler.triggerForAll();
}
