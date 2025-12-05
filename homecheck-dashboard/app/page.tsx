import ProjectsTable from "@/components/ProjectsTable";
import { fetchProjectSummaries } from "@/lib/repository";

export default function Home() {
  const projects = fetchProjectSummaries();
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <main className="mx-auto w-11/12 max-w-[1600px] px-6 py-10">
        <header className="mb-8 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">
              HomeCheck
            </p>
            <h1 className="text-3xl font-semibold text-slate-900">
              缺陷巡检面板
            </h1>
            <p className="text-sm text-slate-500">
              已导入 {projects.length.toLocaleString()} 个项目的 CodeLinter
              结果，可筛选感兴趣的目标后进入详情页进行人工核查。
            </p>
          </div>
          <div className="text-sm text-slate-500">
            数据库文件：
            <code className="ml-2 rounded bg-slate-100 px-2 py-1 text-xs">
              data/homecheck.sqlite
            </code>
          </div>
        </header>
        <ProjectsTable projects={projects} />
      </main>
    </div>
  );
}
