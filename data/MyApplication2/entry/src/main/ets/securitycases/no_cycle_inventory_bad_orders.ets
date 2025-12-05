import { InventoryVendorConnector } from './no_cycle_inventory_bad_vendors';

export class InventoryOrderPlanner {
  private pendingSkus: Array<string> = [];

  constructor(private connector: InventoryVendorConnector) {}

  scheduleRestock(sku: string, qty: number): void {
    this.pendingSkus.push(`${sku}:${qty}`);
    this.connector.requestStock(sku, qty, this);
  }

  finalizeDispatch(): number {
    return this.pendingSkus.length;
  }
}

export const legacyOrderPlanner: InventoryOrderPlanner = new InventoryOrderPlanner(
  new InventoryVendorConnector()
);
legacyOrderPlanner.scheduleRestock('sensor-kit', 12);
