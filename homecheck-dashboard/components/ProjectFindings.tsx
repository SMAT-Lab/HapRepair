'use client';

import { useEffect, useMemo, useState } from "react";
import type {
  FindingRecord,
  ProjectSummary,
} from "@/lib/repository";

const STATUS_OPTIONS = [
  { value: "all", label: "全部状态" },
  { value: "unreviewed", label: "待核查" },
  { value: "valid", label: "确认缺陷" },
  { value: "false_positive", label: "误报" },
  { value: "fixed", label: "已修复" },
] as const;

const CATEGORY_OPTIONS = [
  { value: "all", label: "全部类别" },
  { value: "performance", label: "性能" },
  { value: "security", label: "安全" },
] as const;

const STATUS_COLORS: Record<string, string> = {
  unreviewed: "bg-amber-100 text-amber-700",
  valid: "bg-emerald-100 text-emerald-700",
  false_positive: "bg-slate-200 text-slate-700",
  fixed: "bg-indigo-100 text-indigo-700",
};

const SEVERITY_COLORS: Record<string, string> = {
  error: "text-rose-600",
  warn: "text-amber-600",
  suggestion: "text-slate-500",
};

interface Props {
  project: ProjectSummary;
}

type TreeNode = {
  name: string;
  path: string; // relative to project root
  type: "file" | "dir";
  children?: TreeNode[];
};

export default function ProjectFindings({ project }: Props) {
  const [findings, setFindings] = useState<FindingRecord[]>([]);
  const [, setLoading] = useState(true);
  const [, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] =
    useState<(typeof STATUS_OPTIONS)[number]["value"]>("all");
  const [categoryFilter, setCategoryFilter] =
    useState<(typeof CATEGORY_OPTIONS)[number]["value"]>("all");
  const [search, setSearch] = useState("");
  const [, setUpdatingId] = useState<number | null>(null);
  const [refreshIndex, setRefreshIndex] = useState(0);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [selectedFindingId, setSelectedFindingId] = useState<number | null>(null);
  const [code, setCode] = useState<string | null>(null);
  const [codeLoading, setCodeLoading] = useState(false);
  const [codeError, setCodeError] = useState<string | null>(null);
  const [tree, setTree] = useState<TreeNode[] | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [treeLoading, setTreeLoading] = useState(false);
  const [expandedPaths, setExpandedPaths] = useState<string[]>([]);

  useEffect(() => {
    let canceled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `/api/projects/${project.id}/findings?limit=2000&_=${refreshIndex}`,
          { cache: "no-store" }
        );
        if (!response.ok) {
          throw new Error("无法读取缺陷列表");
        }
        const data = await response.json();
        if (!canceled) {
          setFindings(data.findings ?? []);
        }
      } catch (err) {
        if (!canceled) {
          setError(err instanceof Error ? err.message : "未知错误");
        }
      } finally {
        if (!canceled) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      canceled = true;
    };
  }, [project.id, refreshIndex]);

  useEffect(() => {
    let canceled = false;
    async function loadTree() {
      setTreeLoading(true);
      setTreeError(null);
      try {
        const response = await fetch(`/api/projects/${project.id}/tree`, {
          cache: "no-store",
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "无法读取项目树");
        }
        const data = await response.json();
        if (!canceled) {
          const nodes: TreeNode[] = data.tree ?? [];
          setTree(nodes);
          setExpandedPaths(getDefaultExpandedPaths(nodes));
        }
      } catch (err) {
        if (!canceled) {
          setTreeError(err instanceof Error ? err.message : "无法读取项目树");
        }
      } finally {
        if (!canceled) {
          setTreeLoading(false);
        }
      }
    }
    void loadTree();
    return () => {
      canceled = true;
    };
  }, [project.id]);

  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = {
      unreviewed: 0,
      valid: 0,
      false_positive: 0,
      fixed: 0,
    };
    for (const finding of findings) {
      counts[finding.status] = (counts[finding.status] ?? 0) + 1;
    }
    return counts;
  }, [findings]);

  const filteredFindings = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    return findings.filter((finding) => {
      if (statusFilter !== "all" && finding.status !== statusFilter) {
        return false;
      }
      if (categoryFilter !== "all" && finding.category !== categoryFilter) {
        return false;
      }
      if (!keyword) {
        return true;
      }
      return (
        finding.file_path.toLowerCase().includes(keyword) ||
        finding.rule_id.toLowerCase().includes(keyword) ||
        finding.message.toLowerCase().includes(keyword)
      );
    });
  }, [findings, statusFilter, categoryFilter, search]);

  const fileIssueCounts = useMemo(() => {
    const groups = new Map<string, number>();
    for (const finding of filteredFindings) {
      const key = relativePath(project.root_path, finding.file_path);
      groups.set(key, (groups.get(key) ?? 0) + 1);
    }
    return groups;
  }, [filteredFindings, project.root_path]);

  const currentFileFindings = useMemo(
    () =>
      selectedFile
        ? filteredFindings.filter(
            (finding) =>
              relativePath(project.root_path, finding.file_path) ===
              selectedFile
          )
        : [],
    [filteredFindings, selectedFile, project.root_path]
  );

  const findingsByLine = useMemo(() => {
    const map = new Map<number, FindingRecord[]>();
    for (const finding of currentFileFindings) {
      if (!finding.line) continue;
      const arr = map.get(finding.line) ?? [];
      arr.push(finding);
      map.set(finding.line, arr);
    }
    return map;
  }, [currentFileFindings]);

  const handleStatusChange = async (
    finding: FindingRecord,
    nextStatus: FindingRecord["status"]
  ) => {
    if (finding.status === nextStatus) {
      return;
    }
    await mutateFinding(finding.id, { status: nextStatus });
  };

  const mutateFinding = async (
    id: number,
    payload: Partial<Pick<FindingRecord, "status" | "notes">>
  ) => {
    setError(null);
    try {
      const response = await fetch(`/api/findings/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "更新失败");
      }
      const data = await response.json();
      setFindings((prev) =>
        prev.map((item) => (item.id === id ? data.finding ?? item : item))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新失败");
    } finally {
      setUpdatingId(null);
    }
  };

  const handleSelectFile = (filePath: string) => {
    setSelectedFile(filePath);
    const firstFinding = findings.find(
      (item) =>
        relativePath(project.root_path, item.file_path) === filePath &&
        item.line != null
    );
    setSelectedLine(firstFinding?.line ?? null);
    setSelectedFindingId(firstFinding?.id ?? null);
    void loadCode(filePath);
  };

  const loadCode = async (filePath: string) => {
    setCodeLoading(true);
    setCodeError(null);
    try {
      const response = await fetch(
        `/api/projects/${project.id}/file?path=${encodeURIComponent(filePath)}`,
        { cache: "no-store" }
      );
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "无法读取文件内容");
      }
      const data = await response.json();
      setCode(data.content ?? "");
    } catch (err) {
      setCode("");
      setCodeError(err instanceof Error ? err.message : "无法读取文件内容");
    } finally {
      setCodeLoading(false);
    }
  };

  // 只展示以 entry/src/main/ets 为根的子树（若存在），否则展示整个树
  const viewTree = useMemo(() => {
    if (!tree) return null;
    const target = pickSubtree(tree, "entry/src/main/ets");
    if (target) {
      return [target];
    }
    return tree;
  }, [tree]);

  return (
    <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
      <div className="border-b border-slate-100 px-6 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-slate-800">项目视图</h2>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
              告警 {findings.length.toLocaleString()} 条
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:ml-4">
            {STATUS_OPTIONS.slice(1).map((option) => (
              <span
                key={option.value}
                className={`rounded-full px-3 py-1 text-xs font-medium ${
                  STATUS_COLORS[option.value] ?? "bg-slate-100 text-slate-600"
                }`}
              >
                {option.label}：{statusCounts[option.value]?.toLocaleString() ?? 0}
              </span>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-3">
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="按路径 / 规则 / 描述搜索…"
              className="w-40 rounded-xl border border-slate-200 px-3 py-1.5 text-xs focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-200 sm:w-56"
            />
            <select
              value={statusFilter}
              onChange={(event) =>
                setStatusFilter(
                  event.target.value as (typeof STATUS_OPTIONS)[number]["value"]
                )
              }
              className="rounded-xl border border-slate-200 px-3 py-1.5 text-xs text-slate-700 focus:border-indigo-500 focus:outline-none"
            >
              {STATUS_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <select
              value={categoryFilter}
              onChange={(event) =>
                setCategoryFilter(
                  event.target.value as (typeof CATEGORY_OPTIONS)[number]["value"]
                )
              }
              className="rounded-xl border border-slate-200 px-3 py-1.5 text-xs text-slate-700 focus:border-indigo-500 focus:outline-none"
            >
              {CATEGORY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setRefreshIndex((value) => value + 1)}
              className="rounded-xl border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:border-indigo-300 hover:text-indigo-600"
            >
              重新加载
            </button>
          </div>
        </div>
      </div>
      <div className="relative flex gap-4 border-t border-slate-100 px-4 pb-4 pt-3">
        <div className="flex min-h-[260px] w-64 flex-none flex-col rounded-2xl border border-slate-100 bg-slate-50/80">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2">
            <div className="flex flex-col">
              <p className="text-xs font-semibold text-slate-700">项目树</p>
              <p className="text-xs text-slate-500">
                展示代码文件结构，并标注其中包含告警的文件。
              </p>
            </div>
          </div>
          <div className="flex-1 overflow-auto p-3">
            {treeLoading ? (
              <p className="px-2 py-2 text-xs text-slate-500">正在加载项目树…</p>
            ) : treeError ? (
              <p className="px-2 py-2 text-xs text-rose-500">{treeError}</p>
            ) : !viewTree || viewTree.length === 0 ? (
              <p className="px-2 py-2 text-xs text-slate-400">暂无项目文件</p>
            ) : (
              <ul className="space-y-1 text-xs text-slate-700">
                {decorateTree(viewTree, fileIssueCounts).map((node) => (
                  <TreeNodeView
                    key={node.path || node.name}
                    node={node}
                    level={0}
                    expandedPaths={expandedPaths}
                    onToggle={(path) =>
                      setExpandedPaths((prev) =>
                        prev.includes(path)
                          ? prev.filter((p) => p !== path)
                          : [...prev, path]
                      )
                    }
                    onSelectFile={handleSelectFile}
                    selectedFile={selectedFile}
                  />
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="flex min-h-[260px] flex-1 flex-col rounded-2xl border border-slate-100 bg-slate-50/60">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2">
            <div className="flex flex-col">
              <p className="text-xs font-semibold text-slate-700">代码预览</p>
              <p className="text-xs text-slate-500">
                在左侧项目树中选择文件，这里展示代码与行级告警。
              </p>
            </div>
          </div>
          <div className="relative flex flex-1 gap-3 rounded-b-2xl">
            <div className="relative flex min-h-[220px] flex-1 flex-col rounded-xl border border-slate-200 bg-slate-950/95 text-xs text-slate-100">
              <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
                <p className="truncate font-mono text-[11px] text-slate-300">
                  {selectedFile
                    ? relativePath(project.root_path, selectedFile)
                    : "未选择文件"}
                </p>
                {selectedLine ? (
                  <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-200">
                    行 {selectedLine}
                  </span>
                ) : null}
              </div>
              <div className="relative flex-1 overflow-auto">
                {codeLoading ? (
                  <p className="px-3 py-3 text-[11px] text-slate-300">
                    正在加载代码…
                  </p>
                ) : codeError ? (
                  <p className="px-3 py-3 text-[11px] text-rose-300">
                    {codeError}
                  </p>
                ) : !code || !selectedFile ? (
                  <p className="px-3 py-3 text-[11px] text-slate-400">
                    尚未选择文件。
                  </p>
                ) : (
                  <pre className="px-3 py-3 text-[11px] leading-5">
                    {code.split("\n").map((lineText, index) => {
                      const lineNo = index + 1;
                      const findingsAtLine = findingsByLine.get(lineNo) ?? [];
                      const hasFinding = findingsAtLine.length > 0;
                      const isActiveLine =
                        selectedLine != null && selectedLine === lineNo;
                      return (
                        <div key={lineNo} className="flex flex-col">
                          <div
                            className={`flex gap-3 ${
                              isActiveLine
                                ? "bg-indigo-900/60"
                                : hasFinding
                                  ? "bg-slate-900/60"
                                  : ""
                            }`}
                          >
                            <span className="w-10 shrink-0 text-right text-[10px] text-slate-500">
                              {lineNo}
                            </span>
                            <span className="whitespace-pre text-slate-100">
                              {lineText || " "}
                            </span>
                          </div>
                          {findingsAtLine.length > 0 && (
                            <div className="flex flex-wrap gap-1 bg-slate-900/80 px-10 py-1">
                              {findingsAtLine.map((finding) => (
                                <button
                                  key={finding.id}
                                  type="button"
                                  className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] text-slate-100 hover:border-indigo-400 hover:text-indigo-100 ${
                                    selectedFindingId === finding.id
                                      ? "border-indigo-400 bg-slate-800"
                                      : "border-slate-700 bg-slate-800"
                                  }`}
                                  onClick={() => {
                                    setSelectedFindingId(finding.id);
                                    setSelectedLine(lineNo);
                                  }}
                                >
                                  <span
                                    className={`font-semibold ${
                                      SEVERITY_COLORS[finding.severity] ??
                                      "text-slate-200"
                                    }`}
                                  >
                                    {finding.severity[0].toUpperCase()}
                                  </span>
                                  <span className="max-w-[140px] truncate">
                                    @{finding.category}/{finding.rule_id}
                                  </span>
                                  <span className="rounded bg-slate-700 px-1 text-[9px] uppercase text-slate-200">
                                    {finding.status === "unreviewed"
                                      ? "待核查"
                                      : finding.status === "valid"
                                        ? "确认"
                                        : finding.status === "false_positive"
                                          ? "误报"
                                          : "已修复"}
                                  </span>
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </pre>
                )}
              </div>
            </div>
            <div className="flex w-80 flex-none">
              <FindingDetailsPanel
                finding={
                  selectedFindingId != null
                    ? findings.find((f) => f.id === selectedFindingId) ?? null
                    : null
                }
                code={code}
                onChangeStatus={(finding, status) =>
                  handleStatusChange(finding, status)
                }
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function relativePath(root: string | null, filePath: string) {
  if (!root) {
    return filePath;
  }
  if (filePath.startsWith(root)) {
    return filePath.slice(root.length).replace(/^\//, "") || filePath;
  }
  return filePath;
}

function decorateTree(
  nodes: TreeNode[],
  issueCounts: Map<string, number>
): DecoratedTreeNode[] {
  return nodes.map<DecoratedTreeNode>((node) => {
    if (node.type === "file") {
      return {
        ...node,
        issueCount: issueCounts.get(node.path) ?? 0,
      };
    }
    const rawChildren = node.children ?? [];
    const children = decorateTree(rawChildren, issueCounts);
    const issueCount = children.reduce(
      (sum, child) => sum + child.issueCount,
      0
    );
    return {
      ...node,
      children,
      issueCount,
    };
  });
}

function pickSubtree(nodes: TreeNode[], targetPath: string): TreeNode | null {
  for (const node of nodes) {
    if (node.path === targetPath) {
      return node;
    }
    if (node.type === "dir" && node.children && node.children.length > 0) {
      const found = pickSubtree(node.children, targetPath);
      if (found) {
        return found;
      }
    }
  }
  return null;
}

function getDefaultExpandedPaths(tree: TreeNode[]): string[] {
  const expanded = new Set<string>();

  const findNodeByPath = (nodes: TreeNode[], target: string): TreeNode | null => {
    for (const node of nodes) {
      if (node.path === target) {
        return node;
      }
      if (node.type === "dir" && node.children && node.children.length > 0) {
        const found = findNodeByPath(node.children, target);
        if (found) {
          return found;
        }
      }
    }
    return null;
  };

  const pathsToRoot = ["entry", "entry/src", "entry/src/main", "entry/src/main/ets"];
  let baseNode: TreeNode | null = null;
  for (const p of pathsToRoot) {
    const node = findNodeByPath(tree, p);
    if (node) {
      expanded.add(p);
      baseNode = node;
    }
  }

  if (!baseNode || !baseNode.children) {
    return Array.from(expanded);
  }

  const walk = (nodes: TreeNode[], depth: number, maxDepth: number) => {
    if (depth > maxDepth) return;
    for (const node of nodes) {
      if (node.type === "dir") {
        expanded.add(node.path);
        if (node.children && node.children.length > 0) {
          walk(node.children, depth + 1, maxDepth);
        }
      }
    }
  };

  // 展开 entry/src/main/ets 下两层目录
  walk(baseNode.children, 1, 2);

  return Array.from(expanded);
}

interface TreeNodeViewProps {
  node: DecoratedTreeNode;
  level: number;
  expandedPaths: string[];
  onToggle: (path: string) => void;
  onSelectFile: (path: string) => void;
  selectedFile: string | null;
}

function TreeNodeView({
  node,
  level,
  expandedPaths,
  onToggle,
  onSelectFile,
  selectedFile,
}: TreeNodeViewProps) {
  const isDir = node.type === "dir";
  const isExpanded = expandedPaths.includes(node.path);
  const hasChildren = !!node.children && node.children.length > 0;
  const issueCount = node.issueCount;
  const isSelected = !isDir && selectedFile === node.path;

  const indentStyle = { paddingLeft: `${level * 0.75}rem` };

  return (
    <>
      <li>
        <div
          className={`flex items-center justify-between rounded-lg px-2 py-1 ${
            isSelected
              ? "bg-indigo-50 text-indigo-700"
              : issueCount > 0
                ? "bg-amber-50 text-slate-800"
                : "hover:bg-slate-100"
          }`}
          style={indentStyle}
        >
          <button
            type="button"
            className="flex flex-1 items-center gap-2 text-left"
            onClick={() =>
              isDir ? onToggle(node.path) : onSelectFile(node.path)
            }
          >
            <span className="text-[10px] text-slate-500">
              {isDir ? (isExpanded ? "▾" : "▸") : "•"}
            </span>
            <span className="truncate">{node.name}</span>
          </button>
          {issueCount > 0 && (
            <span className="ml-2 shrink-0 rounded-full bg-rose-50 px-2 py-0.5 text-[10px] font-medium text-rose-500">
              {issueCount}
            </span>
          )}
        </div>
      </li>
      {isDir && isExpanded && hasChildren && (
        <li>
            <ul className="space-y-1">
            {(node.children ?? []).map((child) => (
              <TreeNodeView
                key={child.path || child.name}
                node={child}
                level={level + 1}
                expandedPaths={expandedPaths}
                onToggle={onToggle}
                onSelectFile={onSelectFile}
                selectedFile={selectedFile}
              />
            ))}
          </ul>
        </li>
      )}
    </>
  );
}

interface FindingDetailsPanelProps {
  finding: FindingRecord | null;
  code: string | null;
  onChangeStatus: (finding: FindingRecord, status: FindingRecord["status"]) => void;
}

function FindingDetailsPanel({
  finding,
  code,
  onChangeStatus,
}: FindingDetailsPanelProps) {
  const [ragExamples, setRagExamples] = useState<
    {
      id: string;
      score: number;
      rule: string;
      description: string;
      problem_code: string;
      fix_code: string;
    }[]
  >([]);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState<string | null>(null);

  useEffect(() => {
    if (!finding || !code) {
      setRagExamples([]);
      return;
    }
    let canceled = false;
    async function loadRag() {
      setRagLoading(true);
      setRagError(null);
      try {
        const resp = await fetch("/api/rag/examples", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rule: finding.rule_id,
            code,
            top_k: 3,
          }),
        });
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(text || "RAG 请求失败");
        }
        const data = await resp.json();
        if (!canceled) {
          setRagExamples(data.examples ?? []);
        }
      } catch (err) {
        if (!canceled) {
          setRagError(err instanceof Error ? err.message : "RAG 请求失败");
        }
      } finally {
        if (!canceled) {
          setRagLoading(false);
        }
      }
    }
    void loadRag();
    return () => {
      canceled = true;
    };
  }, [finding, code]);

  if (!finding) {
    return (
      <div className="flex h-full items-center justify-center rounded-xl border border-slate-200 bg-slate-100 px-3 py-3 text-[11px] text-slate-500">
        当前未选中任何告警。
        <br />
        请在代码视图中点击某行下方的标签以查看详情。
      </div>
    );
  }

  const statusLabel =
    finding.status === "unreviewed"
      ? "待核查"
      : finding.status === "valid"
        ? "确认缺陷"
        : finding.status === "false_positive"
          ? "误报"
          : "已修复";

  return (
    <div className="flex h-full flex-col rounded-xl border border-slate-800 bg-slate-900/95 px-3 py-3 text-[11px] leading-5 text-slate-100">
      <div className="flex flex-col gap-2">
        <div>
          <p className="text-[10px] uppercase text-slate-400">规则</p>
          <p className="mt-0.5 break-all font-mono text-[11px]">
            @{finding.category}/{finding.rule_id}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] ${
              SEVERITY_COLORS[finding.severity] ?? "text-slate-200"
            }`}
          >
            {finding.severity.toUpperCase()}
          </span>
          <select
            className="rounded-full border border-slate-600 bg-slate-800 px-2 py-0.5 text-[10px] text-slate-100 focus:border-indigo-400 focus:outline-none"
            value={finding.status}
            onChange={(event) =>
              onChangeStatus(
                finding,
                event.target.value as FindingRecord["status"]
              )
            }
          >
            <option value="unreviewed">待核查</option>
            <option value="valid">确认缺陷</option>
            <option value="false_positive">误报</option>
            <option value="fixed">已修复</option>
          </select>
        </div>
      </div>
      <div className="mt-3 space-y-2">
        <div>
          <p className="text-[10px] uppercase text-slate-400">位置</p>
          <p className="mt-0.5 break-all text-[11px] text-slate-200">
            {finding.file_path}
            <br />
            行 {finding.line ?? "-"} 列 {finding.column ?? "-"}
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase text-slate-400">状态</p>
          <p className="mt-0.5 text-[11px] text-slate-200">{statusLabel}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase text-slate-400">描述</p>
          <p className="mt-0.5 break-words text-[11px] text-slate-100">
            {finding.message}
          </p>
        </div>
      </div>
      <div className="mt-3 border-t border-slate-700 pt-2">
        <p className="text-[10px] uppercase text-slate-400">相关示例（RAG）</p>
        {ragLoading ? (
          <p className="mt-1 text-[11px] text-slate-400">正在检索示例…</p>
        ) : ragError ? (
          <p className="mt-1 break-words text-[11px] text-rose-300">
            {ragError}
          </p>
        ) : ragExamples.length === 0 ? (
          <p className="mt-1 text-[11px] text-slate-400">暂无示例。</p>
        ) : (
          <div className="mt-1 space-y-2">
            {ragExamples.map((ex, index) => (
              <div
                key={ex.id}
                className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5"
              >
                <p className="mb-1 flex items-center justify-between text-[10px] text-slate-300">
                  <span>示例 {index + 1}</span>
                  <span className="text-[9px] text-slate-500">
                    相似度 {(ex.score * 100).toFixed(1)}%
                  </span>
                </p>
                {ex.description && (
                  <p className="mb-1 line-clamp-2 text-[10px] text-slate-300">
                    {ex.description}
                  </p>
                )}
                {ex.problem_code && (
                  <div className="mb-1">
                    <p className="text-[9px] text-rose-200">问题代码</p>
                    <pre className="max-h-20 overflow-auto rounded bg-slate-950 px-2 py-1 text-[10px] text-rose-100">
                      {ex.problem_code}
                    </pre>
                  </div>
                )}
                {ex.fix_code && (
                  <div>
                    <p className="text-[9px] text-emerald-200">修复代码</p>
                    <pre className="max-h-20 overflow-auto rounded bg-slate-950 px-2 py-1 text-[10px] text-emerald-100">
                      {ex.fix_code}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
