#!/usr/bin/env tsx
import fs from "node:fs";
import path from "node:path";
import {
  saveProjectSummary,
  replaceProjectFindings,
  type NewFindingInput,
} from "../lib/repository";

const ANSI_REGEX = /\x1b\[[0-9;]*m/g;
const FILE_HEADER_REGEX = /^(\/.+)\(\d+\)$/;
const VALID_CATEGORIES = new Set(["performance", "security"]);

interface ParseResult {
  slug: string;
  projectName: string;
  rootPath: string | null;
  logPath: string;
  findings: NewFindingInput[];
  severityCounts: Record<string, number>;
  categoryCounts: Record<string, number>;
}

function usage(): never {
  console.error("Usage: tsx scripts/import-logs.ts --log-dir <path> [--limit N]");
  process.exit(1);
}

function parseArgs() {
  const args = process.argv.slice(2);
  const opts: { logDir?: string; limit?: number } = {};
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--log-dir" && args[i + 1]) {
      opts.logDir = args[++i];
    } else if (arg === "--limit" && args[i + 1]) {
      opts.limit = Number(args[++i]);
    } else if (arg === "--help") {
      usage();
    }
  }
  if (!opts.logDir) {
    usage();
  }
  return opts;
}

function sanitize(line: string): string {
  return line.replace(ANSI_REGEX, "").trimEnd();
}

function parseLog(logPath: string): ParseResult | null {
  const slug = path.basename(logPath).replace(/\.log$/i, "");
  const projectName = slug;
  const raw = fs.readFileSync(logPath, "utf8");
  const lines = raw.split(/\r?\n/).map(sanitize);
  let currentFile: string | null = null;
  const files: string[] = [];
  const findings: NewFindingInput[] = [];
  const severityCounts: Record<string, number> = {
    error: 0,
    warn: 0,
    suggestion: 0,
  };
  const categoryCounts: Record<string, number> = {
    performance: 0,
    security: 0,
  };

  for (const line of lines) {
    if (!line) {
      continue;
    }

    const header = line.match(FILE_HEADER_REGEX);
    if (header) {
      currentFile = header[1];
      files.push(currentFile);
      continue;
    }

    if (!currentFile) {
      continue;
    }

    const atIndex = line.lastIndexOf("@");
    if (atIndex === -1) {
      continue;
    }

    const prefix = line.slice(0, atIndex).trim();
    const meta = line.slice(atIndex + 1).trim();
    const match = prefix.match(/^(\d+):(\d+)\s+(error|warn|warning|suggestion)\s+(.*)$/i);
    if (!match) {
      continue;
    }

    const [, lineNum, columnNum, severityRaw, message] = match;
    const [categoryRaw, ruleRaw] = meta.split("/");
    const category = categoryRaw?.trim().toLowerCase() ?? "";
    const severity = normalizeSeverity(severityRaw);
    if (!ruleRaw) {
      continue;
    }
    const ruleId = ruleRaw.trim();
    if (!VALID_CATEGORIES.has(category)) {
      continue;
    }
    findings.push({
      file_path: currentFile,
      line: Number(lineNum),
      column: Number(columnNum),
      severity,
      category,
      rule_id: ruleId,
      message: message.trim(),
    });
    severityCounts[severity] = (severityCounts[severity] ?? 0) + 1;
    categoryCounts[category] = (categoryCounts[category] ?? 0) + 1;
  }

  const rootPath =
    files.length > 0 ? deriveRootPath(files[0], projectName) ?? path.dirname(files[0]) : null;

  return {
    slug,
    projectName,
    rootPath,
    logPath,
    findings,
    severityCounts,
    categoryCounts,
  };
}

function normalizeSeverity(value: string): "error" | "warn" | "suggestion" {
  const normalized = value.trim().toLowerCase();
  if (normalized === "warning") {
    return "warn";
  }
  if (normalized === "warn" || normalized === "error") {
    return normalized as "warn" | "error";
  }
  return "suggestion";
}

function deriveRootPath(filePath: string, projectName: string): string | null {
  const idx = filePath.indexOf(projectName);
  if (idx === -1) {
    return null;
  }
  return filePath.slice(0, idx + projectName.length);
}

async function main() {
  const { logDir, limit } = parseArgs();
  const dir = path.resolve(logDir!);
  const entries = fs
    .readdirSync(dir)
    .filter((name) => name.toLowerCase().endsWith(".log"))
    .slice(0, limit ?? undefined);

  console.log(`Found ${entries.length} log files in ${dir}.`);

  for (const name of entries) {
    const logPath = path.join(dir, name);
    try {
      const parsed = parseLog(logPath);
      if (!parsed) {
        console.warn(`[skip] Could not parse ${logPath}`);
        continue;
      }
      const { slug, projectName, rootPath, findings, severityCounts, categoryCounts } = parsed;
      const project = saveProjectSummary({
        name: projectName,
        slug,
        root_path: rootPath ?? "",
        log_path: logPath,
        total_defects: findings.length,
        perf_defects: categoryCounts.performance ?? 0,
        security_defects: categoryCounts.security ?? 0,
        error_count: severityCounts.error ?? 0,
        warn_count: severityCounts.warn ?? 0,
        suggestion_count: severityCounts.suggestion ?? 0,
      });
      replaceProjectFindings(project.id, findings);
      console.log(`[ok] Imported ${findings.length} findings for ${project.name}`);
    } catch (error) {
      console.error(`[fail] ${logPath}`, error);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
