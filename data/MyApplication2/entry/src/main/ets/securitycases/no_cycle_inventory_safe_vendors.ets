export interface RestockRequest {
  sku: string;
  qty: number;
}

export class InventoryVendorGateway {
  enqueueRequest(request: RestockRequest, onComplete: (token: string) => string): string {
    const tracking = `${request.sku}-${request.qty}`;
    onComplete(tracking);
    return tracking;
  }
}
