'use client';

import Link from "next/link";
import { useMemo, useState } from "react";
import type { ProjectSummary } from "@/lib/repository";

const PAGE_SIZE_OPTIONS = [25, 50, 100] as const;

type SortColumn =
  | "name"
  | "total_defects"
  | "perf_defects"
  | "security_defects"
  | "error_count"
  | "warn_count"
  | "suggestion_count";

interface SortState {
  column: SortColumn;
  direction: "asc" | "desc";
}

interface Props {
  projects: ProjectSummary[];
}

export default function ProjectsTable({ projects }: Props) {
  const [query, setQuery] = useState("");
  const [securityOnly, setSecurityOnly] = useState(false);
  const [pageSize, setPageSize] = useState<(typeof PAGE_SIZE_OPTIONS)[number]>(
    PAGE_SIZE_OPTIONS[1]
  );
  const filtersKey = `${query}|${securityOnly}|${pageSize}`;
  const [pageState, setPageState] = useState(() => ({
    page: 1,
    filtersKey,
  }));
  const [sortState, setSortState] = useState<SortState>({
    column: "total_defects",
    direction: "desc",
  });

  const sorted = useMemo(() => {
    return [...projects].sort((a, b) => {
      const { column, direction } = sortState;
      const fallback = a.name.localeCompare(b.name);
      const compareResult = compareProjectValues(a, b, column);
      if (compareResult === 0) {
        return fallback;
      }
      return direction === "asc" ? compareResult : -compareResult;
    });
  }, [projects, sortState]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return sorted.filter((project) => {
      if (securityOnly && project.security_defects === 0) {
        return false;
      }
      if (!q) {
        return true;
      }
      return (
        project.name.toLowerCase().includes(q) ||
        project.slug.toLowerCase().includes(q) ||
        project.root_path.toLowerCase().includes(q)
      );
    });
  }, [sorted, query, securityOnly]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const activePage =
    pageState.filtersKey === filtersKey ? pageState.page : 1;
  const currentPage = Math.max(1, Math.min(activePage, totalPages));
  const pageItems = filtered.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const goToPage = (next: number) => {
    const normalized = Math.max(1, Math.min(totalPages, next));
    setPageState({ page: normalized, filtersKey });
  };

  const handleSort = (column: SortColumn) => {
    setSortState((prev) => {
      if (prev.column === column) {
        return {
          column,
          direction: prev.direction === "asc" ? "desc" : "asc",
        };
      }
      return { column, direction: "desc" };
    });
  };

  return (
    <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
      <div className="flex flex-col gap-4 border-b border-slate-100 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-1 items-center gap-3">
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索项目或路径…"
            className="w-full rounded-xl border border-slate-200 px-4 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 sm:max-w-sm"
          />
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={securityOnly}
              onChange={(event) => setSecurityOnly(event.target.checked)}
              className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            仅显示安全缺陷
          </label>
        </div>
        <p className="text-xs text-slate-500">
          显示 {filtered.length.toLocaleString()} /{" "}
          {projects.length.toLocaleString()} 个项目
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-100 text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <SortableHeader
                label="项目"
                column="name"
                sortState={sortState}
                onSort={handleSort}
                alignment="left"
              />
              <SortableHeader
                label="总缺陷"
                column="total_defects"
                sortState={sortState}
                onSort={handleSort}
              />
              <SortableHeader
                label="性能"
                column="perf_defects"
                sortState={sortState}
                onSort={handleSort}
              />
              <SortableHeader
                label="安全"
                column="security_defects"
                sortState={sortState}
                onSort={handleSort}
              />
              <SortableHeader
                label="错误"
                column="error_count"
                sortState={sortState}
                onSort={handleSort}
              />
              <SortableHeader
                label="警告"
                column="warn_count"
                sortState={sortState}
                onSort={handleSort}
              />
              <SortableHeader
                label="建议"
                column="suggestion_count"
                sortState={sortState}
                onSort={handleSort}
              />
              <th className="px-4 py-3 text-left font-semibold">最近一次扫描</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {pageItems.map((project) => (
              <tr key={project.id} className="hover:bg-slate-50">
                <td className="px-4 py-3">
                  <div className="flex flex-col">
                    <Link
                      href={`/projects/${encodeURIComponent(project.slug)}`}
                      className="font-medium text-indigo-600 hover:underline"
                    >
                      {project.name}
                    </Link>
                    <span className="text-xs text-slate-500">
                      {project.root_path || "路径未知"}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right font-medium text-slate-900">
                  {project.total_defects.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right text-slate-700">
                  {project.perf_defects.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right text-rose-600">
                  {project.security_defects.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right">
                  {project.error_count.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right">
                  {project.warn_count.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right">
                  {project.suggestion_count.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-left text-xs text-slate-500">
                  {formatDate(project.last_run)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-col gap-4 border-t border-slate-100 px-6 py-4 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage === 1}
            className="rounded-lg border border-slate-200 px-3 py-1 disabled:opacity-40"
          >
            上一页
          </button>
          <span>
            第 {currentPage}/{totalPages} 页
          </span>
          <button
            type="button"
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="rounded-lg border border-slate-200 px-3 py-1 disabled:opacity-40"
          >
            下一页
          </button>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm">每页显示</label>
          <select
            value={pageSize}
            onChange={(event) =>
              setPageSize(Number(event.target.value) as (typeof PAGE_SIZE_OPTIONS)[number])
            }
            className="rounded-xl border border-slate-200 px-3 py-1"
          >
            {PAGE_SIZE_OPTIONS.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
          <span className="text-xs text-slate-500">
            当前显示 {pageItems.length} / {filtered.length} （总计{" "}
            {projects.length}）
          </span>
        </div>
      </div>
    </section>
  );
}

function formatDate(value: string) {
  if (!value) {
    return "未知";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function compareProjectValues(
  a: ProjectSummary,
  b: ProjectSummary,
  column: SortColumn
): number {
  if (column === "name") {
    return a.name.localeCompare(b.name);
  }
  const aValue = a[column] ?? 0;
  const bValue = b[column] ?? 0;
  if (aValue === bValue) {
    return 0;
  }
  return aValue > bValue ? 1 : -1;
}

interface SortableHeaderProps {
  label: string;
  column: SortColumn;
  sortState: SortState;
  onSort: (column: SortColumn) => void;
  alignment?: "left" | "right";
}

function SortableHeader({
  label,
  column,
  sortState,
  onSort,
  alignment = "right",
}: SortableHeaderProps) {
  const isActive = sortState.column === column;
  const direction = isActive ? sortState.direction : undefined;
  const icon =
    direction === "asc"
      ? "↑"
      : direction === "desc"
        ? "↓"
        : "↕";
  const alignmentClass = alignment === "left" ? "text-left" : "text-right";

  return (
    <th className={`px-4 py-3 font-semibold ${alignmentClass}`} scope="col">
      <button
        type="button"
        onClick={() => onSort(column)}
        className={`inline-flex items-center gap-1 text-xs uppercase tracking-wide ${
          isActive ? "text-indigo-600" : "text-slate-500"
        }`}
      >
        {label}
        <span aria-hidden="true" className="text-xs">
          {icon}
        </span>
      </button>
    </th>
  );
}
