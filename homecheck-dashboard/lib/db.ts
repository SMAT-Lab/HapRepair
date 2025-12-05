import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";

const dataDir = path.join(process.cwd(), "data");
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

const dbPath = process.env.HOMECHECK_DB ?? path.join(dataDir, "homecheck.sqlite");
const db = new Database(dbPath);
db.pragma("journal_mode = WAL");

db.exec(`
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  root_path TEXT NOT NULL,
  log_path TEXT,
  total_defects INTEGER NOT NULL DEFAULT 0,
  perf_defects INTEGER NOT NULL DEFAULT 0,
  security_defects INTEGER NOT NULL DEFAULT 0,
  error_count INTEGER NOT NULL DEFAULT 0,
  warn_count INTEGER NOT NULL DEFAULT 0,
  suggestion_count INTEGER NOT NULL DEFAULT 0,
  last_run TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  file_path TEXT NOT NULL,
  line INTEGER,
  column INTEGER,
  severity TEXT NOT NULL,
  category TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  message TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unreviewed',
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_findings_project ON findings(project_id);
CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);
CREATE INDEX IF NOT EXISTS idx_findings_rule ON findings(rule_id);
`);

export type FindingStatus = "unreviewed" | "valid" | "false_positive" | "fixed";

export { db };
