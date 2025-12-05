import { InventoryVendorGateway } from './no_cycle_inventory_safe_vendors';

export class InventoryOrderCoordinator {
  private gateway: InventoryVendorGateway;

  constructor() {
    this.gateway = new InventoryVendorGateway();
  }

  scheduleRestock(sku: string, qty: number): number {
    return this.gateway.enqueueRequest({ sku, qty }, (result: string) => {
      return `${result}:${sku}`;
    }).length;
  }
}

export const coordinator: InventoryOrderCoordinator = new InventoryOrderCoordinator();
coordinator.scheduleRestock('sensor-kit', 5);
