#!/usr/bin/env python3
"""
Ingest curated ArkTS repair examples into Pinecone `arkts-1536`, including
problem/fix code, difflib, and gpt_diff (LLM-generated when API key available,
otherwise a concise manual summary).

Usage:
  PINECONE_API_KEY=... python scripts/ingest_rag_examples.py

Notes:
  - Embeddings use the same model dimension (1536) as the repair pipeline:
    dunzhang/stella_en_1.5B_v5.
  - Namespace is fixed to "arkts" to align with get_rag_prompt filtering.
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer  # type: ignore

try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore

INDEX_NAME = "arkts-1536"
NAMESPACE = "arkts"
EMBED_MODEL = "dunzhang/stella_en_1.5B_v5"


@dataclass
class Entry:
    rule: str
    description: str
    problem_code: str
    problem_explain: str
    problem_fix: str
    gpt_diff_manual: str
    id: str


def unified_diff_text(a: str, b: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True)))


def build_gpt_diff(entry: Entry, diff_text: str) -> str:
    """
    Try to generate gpt_diff via OpenAI if OPENAI_API_KEY is set and openai is installed.
    Fallback to the provided manual summary.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or openai is None:
        return entry.gpt_diff_manual

    openai.api_key = api_key  # type: ignore
    prompt = (
        "Given a buggy ArkTS snippet and its fixed version, summarize the key actions "
        "needed to fix it in 2-4 bullet points.\n\n"
        f"Buggy code:\n```arkts\n{entry.problem_code}\n```\n\n"
        f"Fixed code:\n```arkts\n{entry.problem_fix}\n```\n\n"
        "Return only the bullet list in Chinese."
    )
    try:
        resp = openai.ChatCompletion.create(  # type: ignore
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return entry.gpt_diff_manual


def get_entries() -> List[Entry]:
    return [
        Entry(
            id="waterflow-preload-01",
            rule="@performance/waterflow-data-preload-check",
            description="建议在 WaterFlow 的 FlowItem 中预加载数据，使用 DataSource.notifyDataAdd。",
            problem_code="""
WaterFlow() {
  LazyForEach(this.items, (item: number) => {
    FlowItem() {
      Text(`${item}`)
    }
  })
}
.cachedCount(2)
.onReachEnd(() => {
  for (let i = 0; i < 40; i++) {
    this.items.push(this.items.length + i)
  }
})
""",
            problem_explain="只在 onReachEnd 里 push 数组，未在 FlowItem.onAppear 触发 DataSource 的新增通知，未预加载。",
            problem_fix="""
@State dataSource: WaterFlowDataSource = new WaterFlowDataSource();

WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      Text(`${item}`)
    }
    .onAppear(() => {
      if (item + 10 >= this.dataSource.totalCount()) {
        this.dataSource.addLastItem();
      }
    })
  }, (item) => item)
}
.cachedCount(2)

// WaterFlowDataSource implements IDataSource and calls notifyDataAdd in addLastItem().
""",
            gpt_diff_manual="将 WaterFlow 的数据源改为 DataSource，并在 FlowItem.onAppear 里触发 addLastItem/notifyDataAdd，移除 onReachEnd 的延迟加载。",
        ),
        Entry(
            id="avoid-custom-01",
            rule="@performance/avoid-overusing-custom-component-check",
            description="循环中优先使用 @Builder 或可复用组件，减少自定义组件节点。",
            problem_code="""
List() {
  ForEach(this.users, (item: User) => {
    ListItem() {
      UserCard({ name: item.name, age: item.age })
    }
  })
}

@Component
struct UserCard {
  @Prop name: string;
  @Prop age: number;
  build() {
    Row() {
      Text(this.name)
      Text(`${this.age}`)
    }
  }
}
""",
            problem_explain="ForEach 内直接使用自定义组件 UserCard，增加 CustomNode 数量。",
            problem_fix="""
@Builder
function UserCardBuilder(name: string, age: number) {
  Row() {
    Text(name)
    Text(`${age}`)
  }
}

List() {
  ForEach(this.users, (item: User) => {
    ListItem() {
      UserCardBuilder(item.name, item.age)
    }
  }, (item) => item.id)
}
""",
            gpt_diff_manual="将 ForEach 中的自定义组件替换为 @Builder 函数，保留渲染内容并补充稳定 keyGenerator。",
        ),
        Entry(
            id="swiper-preload-01",
            rule="@performance/hp-arkui-use-onAnimationStart-for-swiper-preload",
            description="Swiper 预加载应放在 onAnimationStart 回调中。",
            problem_code="""
Swiper(this.ctrl) {
  ForEach(this.photos, (item: Photo, idx?: number) => {
    Image(item.url)
  }, (item: Photo) => item.id)
}
.cachedCount(1)
.onChange((index: number) => {
  preload(this.photos[index + 1]);
})
""",
            problem_explain="预加载放在 onChange，切换完成才触发，未利用 onAnimationStart。",
            problem_fix="""
Swiper(this.ctrl) {
  ForEach(this.photos, (item: Photo, idx?: number) => {
    Image(item.url)
  }, (item: Photo) => item.id)
}
.cachedCount(1)
.onAnimationStart((index: number, targetIndex: number) => {
  if (targetIndex !== index) {
    preload(this.photos[targetIndex + 1]);
  }
})
""",
            gpt_diff_manual="移除/弱化 onChange 预加载，新增 onAnimationStart 回调，在动画开始阶段预加载下一项。",
        ),
        Entry(
            id="foreach-key-01",
            rule="@performance/foreach-index-check",
            description="ForEach 的 keyGenerator 不应依赖 index，需使用稳定 id。",
            problem_code="""
ForEach(this.items, (item: Item, index: number) => {
  ListItem() {
    Text(item.title)
  }
}, (item: Item, index: number) => `${item.id}-${index}`)
""",
            problem_explain="keyGenerator 拼接 index，插入/删除时会导致大量重建。",
            problem_fix="""
ForEach(this.items, (item: Item, index: number) => {
  ListItem() {
    Text(item.title)
  }
}, (item: Item) => item.id)
""",
            gpt_diff_manual="将 keyGenerator 从含 index 的组合改为稳定的 item.id。",
        ),
        Entry(
            id="dynamic-cls-func-01",
            rule="@performance/hp-performance-no-dynamic-cls-func",
            description="避免在 render/build 中动态定义类或函数。",
            problem_code="""
build() {
  class InlineHelper {
    double(x: number) { return x * 2; }
  }
  const helper = new InlineHelper();
  Column() {
    Text(`${helper.double(this.value)}`)
  }
}
""",
            problem_explain="在 build 中定义类，每次渲染都会创建新类/函数，影响性能。",
            problem_fix="""
class InlineHelper {
  double(x: number) { return x * 2; }
}

build() {
  const helper = new InlineHelper();
  Column() {
    Text(`${helper.double(this.value)}`)
  }
}
""",
            gpt_diff_manual="将动态定义的类移到模块作用域，build 中仅实例化/调用，避免每次渲染创建新函数。",
        ),
    ]


def get_embedding(text: str, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, cache_dir="/home/models")
    model = AutoModel.from_pretrained(EMBED_MODEL, cache_dir="/home/models")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    entries = get_entries()
    vectors = []
    for entry in entries:
        diff_text = unified_diff_text(entry.problem_code, entry.problem_fix)
        gpt_diff = build_gpt_diff(entry, diff_text)
        emb = get_embedding(entry.problem_code, model, tokenizer).tolist()
        vectors.append(
            {
                "id": entry.id,
                "values": emb,
                "metadata": {
                    "rule": entry.rule,
                    "description": entry.description,
                    "problem_code": entry.problem_code.strip(),
                    "problem_explain": entry.problem_explain,
                    "problem_fix": entry.problem_fix.strip(),
                    "gpt_diff": gpt_diff,
                    "difflib": diff_text,
                },
            }
        )

    print(f"Upserting {len(vectors)} entries to index {INDEX_NAME} (namespace={NAMESPACE})...")
    upserted = index.upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Upsert result: {upserted}")


if __name__ == "__main__":
    main()
