import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { updateFindingStatus } from "@/lib/repository";

const statusSchema = z.enum([
  "unreviewed",
  "valid",
  "false_positive",
  "fixed",
] as const);

const updateSchema = z.object({
  status: statusSchema.optional(),
  notes: z.string().max(2000).nullable().optional(),
});

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const findingId = Number(id);
  if (!Number.isFinite(findingId)) {
    return NextResponse.json({ error: "Invalid finding id" }, { status: 400 });
  }

  const payload = await request.json().catch(() => null);
  if (!payload) {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = updateSchema.safeParse(payload);
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.message }, { status: 400 });
  }

  const updated = updateFindingStatus(findingId, parsed.data);
  if (!updated) {
    return NextResponse.json({ error: "Finding not found" }, { status: 404 });
  }

  return NextResponse.json({ finding: updated });
}
