import { db, type FindingStatus } from "./db";

export const FINDING_STATUSES: FindingStatus[] = [
  "unreviewed",
  "valid",
  "false_positive",
  "fixed",
];

export interface ProjectSummary {
  id: number;
  name: string;
  slug: string;
  root_path: string;
  log_path: string | null;
  total_defects: number;
  perf_defects: number;
  security_defects: number;
  error_count: number;
  warn_count: number;
  suggestion_count: number;
  last_run: string;
}

export interface FindingRecord {
  id: number;
  project_id: number;
  file_path: string;
  line: number | null;
  column: number | null;
  severity: string;
  category: string;
  rule_id: string;
  message: string;
  status: FindingStatus;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface SaveProjectInput {
  name: string;
  slug: string;
  root_path: string;
  log_path?: string;
  total_defects: number;
  perf_defects: number;
  security_defects: number;
  error_count: number;
  warn_count: number;
  suggestion_count: number;
  last_run?: string;
}

export interface NewFindingInput {
  file_path: string;
  line?: number;
  column?: number;
  severity: string;
  category: string;
  rule_id: string;
  message: string;
}

const getProjectStmt = db.prepare("SELECT * FROM projects WHERE slug = ?");
const getProjectByIdStmt = db.prepare("SELECT * FROM projects WHERE id = ?");

export function saveProjectSummary(data: SaveProjectInput): ProjectSummary {
  const payload = {
    ...data,
    last_run: data.last_run ?? null,
  };
  const update = db.prepare(
    `UPDATE projects
     SET name=@name,
         root_path=@root_path,
         log_path=COALESCE(@log_path, log_path),
         total_defects=@total_defects,
         perf_defects=@perf_defects,
         security_defects=@security_defects,
         error_count=@error_count,
         warn_count=@warn_count,
         suggestion_count=@suggestion_count,
         last_run=COALESCE(@last_run, CURRENT_TIMESTAMP)
     WHERE slug=@slug`
  );
  const result = update.run(payload);
  if (result.changes === 0) {
    const insert = db.prepare(
      `INSERT INTO projects
      (name, slug, root_path, log_path, total_defects, perf_defects, security_defects,
       error_count, warn_count, suggestion_count, last_run)
      VALUES (@name, @slug, @root_path, @log_path, @total_defects, @perf_defects,
        @security_defects, @error_count, @warn_count, @suggestion_count,
        COALESCE(@last_run, CURRENT_TIMESTAMP))`
    );
    insert.run(payload);
  }
  const row = getProjectStmt.get(data.slug) as ProjectSummary | undefined;
  if (!row) {
    throw new Error(`Project ${data.slug} was not found after upsert.`);
  }
  return row;
}

export function getProjectBySlug(slug: string): ProjectSummary | undefined {
  return getProjectStmt.get(slug) as ProjectSummary | undefined;
}

export function getProjectById(id: number): ProjectSummary | undefined {
  return getProjectByIdStmt.get(id) as ProjectSummary | undefined;
}

export function replaceProjectFindings(projectId: number, findings: NewFindingInput[]): void {
  const insert = db.prepare(
    `INSERT INTO findings
    (project_id, file_path, line, column, severity, category, rule_id, message)
    VALUES (@project_id, @file_path, @line, @column, @severity, @category, @rule_id, @message)`
  );
  const deleteStmt = db.prepare("DELETE FROM findings WHERE project_id = ?");
  const tx = db.transaction(() => {
    deleteStmt.run(projectId);
    for (const finding of findings) {
      insert.run({
        project_id: projectId,
        ...finding,
      });
    }
  });
  tx();
}

export function fetchProjectSummaries(limit?: number): ProjectSummary[] {
  const baseQuery = `SELECT * FROM projects ORDER BY total_defects DESC, name ASC`;
  if (typeof limit === "number" && Number.isFinite(limit) && limit > 0) {
    const stmt = db.prepare(`${baseQuery} LIMIT ?`);
    return stmt.all(limit) as ProjectSummary[];
  }
  const stmt = db.prepare(baseQuery);
  return stmt.all() as ProjectSummary[];
}

export interface FindingFilters {
  status?: FindingStatus;
  category?: string;
  rule?: string;
  search?: string;
  limit?: number;
}

export function fetchFindings(projectId: number, filters: FindingFilters = {}): FindingRecord[] {
  const clauses = ["project_id = ?"];
  const params: Array<string | number> = [projectId];
  if (filters.status) {
    clauses.push("status = ?");
    params.push(filters.status);
  }
  if (filters.category) {
    clauses.push("category = ?");
    params.push(filters.category);
  }
  if (filters.rule) {
    clauses.push("rule_id = ?");
    params.push(filters.rule);
  }
  if (filters.search) {
    clauses.push("(file_path LIKE ? OR message LIKE ?)");
    const pattern = `%${filters.search}%`;
    params.push(pattern, pattern);
  }
  const limit = filters.limit ?? 1000;
  params.push(limit);

  const stmt = db.prepare(
    `SELECT * FROM findings WHERE ${clauses.join(" AND ")}
     ORDER BY severity DESC, created_at DESC
     LIMIT ?`
  );
  return stmt.all(...params) as FindingRecord[];
}

export function updateFindingStatus(
  id: number,
  updates: { status?: FindingStatus; notes?: string | null }
): FindingRecord | undefined {
  const fields: string[] = [];
  const params: Array<string | number | null> = [];
  if (updates.status) {
    fields.push("status = ?");
    params.push(updates.status);
  }
  if (Object.prototype.hasOwnProperty.call(updates, "notes")) {
    fields.push("notes = ?");
    params.push(updates.notes && updates.notes.length > 0 ? updates.notes : null);
  }
  if (fields.length === 0) {
    return getFindingById(id);
  }
  fields.push("updated_at = CURRENT_TIMESTAMP");
  params.push(id);
  const stmt = db.prepare(`UPDATE findings SET ${fields.join(", ")} WHERE id = ?`);
  stmt.run(...params);
  return getFindingById(id);
}

export function getFindingById(id: number): FindingRecord | undefined {
  const stmt = db.prepare("SELECT * FROM findings WHERE id = ?");
  return stmt.get(id) as FindingRecord | undefined;
}
