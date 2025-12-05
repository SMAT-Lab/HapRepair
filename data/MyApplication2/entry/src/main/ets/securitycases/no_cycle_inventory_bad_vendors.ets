import { InventoryOrderPlanner } from './no_cycle_inventory_bad_orders';

export class InventoryVendorConnector {
  requestStock(sku: string, qty: number, planner?: InventoryOrderPlanner): void {
    if (planner) {
      this.logToPlanner(planner, sku);
    }
  }

  private logToPlanner(planner: InventoryOrderPlanner, sku: string): void {
    if (sku.length > 0) {
      planner.finalizeDispatch();
    }
  }
}
