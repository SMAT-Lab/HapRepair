#!/usr/bin/env python3
"""
扫描 data/MyApplication2/entry/src/main/ets/securitycases/ 下的安全规则样例，
自动构建 Excel（data/security_pairs.xlsx），用于向量库构建。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class RuleConfig:
    prefix: str
    rule: str
    description: str


RULE_CONFIGS: List[RuleConfig] = sorted(
    [
        RuleConfig("no_unsafe_dh_key", "@security/no-unsafe-dh-key", "DH 密钥模数必须 >= 2048 bit。"),
        RuleConfig("no_unsafe_dh", "@security/no-unsafe-dh", "DH 协商算法必须使用 2048 bit 以上安全参数。"),
        RuleConfig("no_unsafe_dsa_key", "@security/no-unsafe-dsa-key", "DSA 密钥模数需 >= 2048 bit。"),
        RuleConfig("no_unsafe_dsa", "@security/no-unsafe-dsa", "DSA 签名/验签禁止使用 SHA1，模数需 >= 2048。"),
        RuleConfig("no_unsafe_ecdsa", "@security/no-unsafe-ecdsa", "ECDSA 签名/验签禁止使用 SHA1。"),
        RuleConfig("no_unsafe_ecdh", "@security/no-unsafe-ecdh", "ECC 密钥生成必须选择安全曲线（例如 ECC256 及以上）。"),
        RuleConfig("no_unsafe_rsa_encrypt", "@security/no-unsafe-rsa-encrypt", "RSA 加密需要 2048 bit+ 且使用 OAEP 等安全填充。"),
        RuleConfig("no_unsafe_rsa_key", "@security/no-unsafe-rsa-key", "RSA 密钥长度必须不小于 2048 bit。"),
        RuleConfig("no_unsafe_rsa_sign", "@security/no-unsafe-rsa-sign", "RSA 签名需使用 PSS + SHA256/384 等安全散列。"),
        RuleConfig("no_unsafe_sm2_cipher", "@security/no-unsafe-sm2-cipher", "SM2 加解密禁止使用 MD5/SHA1 等弱摘要。"),
        RuleConfig("no_unsafe_sm2_key", "@security/no-unsafe-sm2-key", "SM2 密钥生成需显式指定安全曲线参数。"),
        RuleConfig("no_unsafe_sm4", "@security/no-unsafe-sm4", "SM4 禁止使用 ECB 等不安全分组模式。"),
        RuleConfig("no_unsafe_huks", "@security/no-unsafe-huks", "HUKS 中禁止 ECB/SHA1/NONE 等不安全配置。"),
        RuleConfig("no_unsafe_aes", "@security/no-unsafe-aes", "AES 只能使用 CBC/GCM + 安全填充，禁止 ECB/NoPadding。"),
        RuleConfig("no_unsafe_3des", "@security/no-unsafe-3des", "3DES 不得使用 ECB 模式。"),
        RuleConfig("no_unsafe_hash", "@security/no-unsafe-hash", "禁止 MD5/SHA1 等弱哈希算法。"),
        RuleConfig("no_unsafe_kdf", "@security/no-unsafe-kdf", "PBKDF2/HKDF 派生不得使用 SHA1。"),
        RuleConfig("no_unsafe_mac", "@security/no-unsafe-mac", "HMAC/MAC 算法禁止使用 SHA1。"),
        RuleConfig("no_commented_code", "@security/no-commented-code", "禁止以注释形式保留废弃或敏感代码。"),
        RuleConfig("no_cycle", "@security/no-cycle", "禁止 ETS 模块之间存在循环依赖。"),
    ],
    key=lambda cfg: len(cfg.prefix),
    reverse=True,
)


def detect_rule(base_name: str) -> Optional[RuleConfig]:
    for cfg in RULE_CONFIGS:
        if base_name.startswith(cfg.prefix):
            return cfg
    return None


def scenario_label(base_name: str, prefix: str) -> str:
    suffix = base_name[len(prefix):].strip("_")
    return suffix.replace("_", " ") if suffix else "默认场景"


def build_security_pairs() -> pd.DataFrame:
    cases_dir = Path("data/MyApplication2/entry/src/main/ets/securitycases")
    records = []

    for bad_path in sorted(cases_dir.glob("*_bad*.ets")):
        base_name = bad_path.stem
        if not base_name.endswith("_bad"):
            continue
        scenario_base = base_name[:-4]  # strip "_bad"
        safe_path = cases_dir / f"{scenario_base}_safe.ets"
        if not safe_path.exists():
            continue

        rule_cfg = detect_rule(scenario_base)
        if not rule_cfg:
            continue

        problem_code = bad_path.read_text(encoding="utf-8").strip()
        repair_code = safe_path.read_text(encoding="utf-8").strip()
        difflib_text = "\n".join(
            unified_diff(
                problem_code.splitlines(),
                repair_code.splitlines(),
                fromfile="problem",
                tofile="fix",
                lineterm="",
            )
        )

        scenario = scenario_label(scenario_base, rule_cfg.prefix)
        desc_core = rule_cfg.description.rstrip("。")
        explanation = (
            f"{scenario} 代码触发 {rule_cfg.rule}：{desc_core}。"
            " 不安全示例保留了 Legacy 配置，修复版本改为推荐配置并保持业务逻辑不变。"
        )
        diff_summary = (
            f"{scenario} 场景下将不安全实现替换为安全写法，以满足 {rule_cfg.rule} 的要求。"
        )

        records.append(
            {
                "Rule": rule_cfg.rule,
                "Description": rule_cfg.description,
                "Problem Code Example": problem_code,
                "Problem Explanation": explanation,
                "Repair Code Example": repair_code,
                "Diff": diff_summary,
                "Difflib": difflib_text,
            }
        )

    df = pd.DataFrame(records)
    df.sort_values(["Rule", "Problem Explanation"], inplace=True, ignore_index=True)
    return df


def main() -> None:
    df = build_security_pairs()
    output_path = Path("data/security_pairs.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Wrote {len(df)} samples to {output_path}")


if __name__ == "__main__":
    main()
