import fs from "node:fs/promises";
import { NextRequest, NextResponse } from "next/server";
import * as XLSX from "xlsx";

interface SecurityExample {
  rule: string;
  description: string;
  problem_code: string;
  repair_code: string;
  explanation?: string;
  diff_summary?: string;
}

let cache: Map<string, SecurityExample[]> | null = null;

async function loadExamples(): Promise<Map<string, SecurityExample[]>> {
  if (cache) {
    return cache;
  }

  const excelPath =
    process.env.SECURITY_PAIRS_PATH ??
    "/home/LLMCodeRepair/data/security_pairs.xlsx";

  const buffer = await fs.readFile(excelPath);
  const workbook = XLSX.read(buffer, { type: "buffer" });
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, {
    defval: "",
  });

  const map = new Map<string, SecurityExample[]>();
  for (const row of rows) {
    const rule = String(row["Rule"] ?? "").trim();
    if (!rule) continue;
    const description = String(row["Description"] ?? "").trim();
    const problem_code = String(row["Problem Code Example"] ?? "").trim();
    const repair_code = String(row["Repair Code Example"] ?? "").trim();
    const explanation = String(row["Problem Explanation"] ?? "").trim();
    const diff_summary = String(row["Diff"] ?? "").trim();

    const example: SecurityExample = {
      rule,
      description,
      problem_code,
      repair_code,
      explanation: explanation || undefined,
      diff_summary: diff_summary || undefined,
    };

    const arr = map.get(rule) ?? [];
    arr.push(example);
    map.set(rule, arr);
  }

  cache = map;
  return map;
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  if (!body) {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { rule, top_k = 3 } = body as {
    rule?: string;
    code?: string;
    top_k?: number;
  };

  if (!rule) {
    return NextResponse.json({ error: "Missing rule" }, { status: 400 });
  }

  try {
    const examplesMap = await loadExamples();
    const fullRule = rule.startsWith("@")
      ? rule
      : rule.includes("/")
        ? `@${rule}`
        : `@security/${rule}`;

    const examples = examplesMap.get(fullRule) ?? [];
    const limited = examples.slice(0, top_k);

    return NextResponse.json({
      examples: limited.map((ex, index) => ({
        id: `${fullRule}#${index}`,
        score: 1.0,
        rule: ex.rule,
        description: ex.explanation || ex.description,
        problem_code: ex.problem_code,
        fix_code: ex.repair_code,
      })),
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          (error as Error).message ??
          "Failed to load local security examples",
      },
      { status: 500 }
    );
  }
}
