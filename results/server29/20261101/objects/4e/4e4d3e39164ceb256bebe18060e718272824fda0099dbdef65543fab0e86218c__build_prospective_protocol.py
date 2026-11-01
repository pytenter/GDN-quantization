"""Write an outcome-blind four-document preregistration; never launch GPU work."""
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
OLD = D.parent / "ling_functional_validation"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, obj):
    (D / name).write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")


spec = read(D / "FROZEN_SPEC.json")
pool = read(D / "PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json")
identities = read(OLD / "evidence/previous/SOURCE_IDENTITIES.json")
sources = read(D / "PROSPECTIVE_FROZEN_SCORE_SOURCE_IDENTITIES.json")
assert pool["pool_books"] == 100 and pool["input_identity_eligible_count"] == 58
assert len(identities["rotations"]) == 16
roster = sorted(identities["rotations"])
eligible = [x for x in pool["inputs"] if x["input_identity_eligible"]]
excluded = pool["historical_excluded_ids"]
rules = dict(
    schema="next_prospective_extension_4doc_preregistration_v1",
    experiment="LING_KDA_P1_L7_PROSPECTIVE_EXTENSION_4DOC_V1",
    status="PREREGISTRATION_AND_CPU_INPUT_IDENTITIES_READY_GPU_EXECUTION_NOT_AUTHORIZED",
    original_formal_verdict="MIXED_PERMANENTLY_PRESERVED",
    future_outcomes_accessed=False, new_document_GPU_science_executed=False,
    complete_candidate_pool=dict(official_source="https://github.com/google-deepmind/pg19",
        split="test", books=100, input_identity_eligible=58, historical_exclusions=20,
        insufficient_length=22, duplicate_count=0,
        full_listing_sha256=sha(D/"PG19_TEST_GCS_LISTING.json"),
        input_identity_audit_sha256=sha(D/"PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json"),
        input_identity_audit_path="PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json",
        object_generations_pinned=True, MD5_and_SHA256_verified=True),
    exclusions=dict(document_ids=excluded,
        registry_path="historical_path_identity_inventory.json",
        registry_sha256=sha(D/"historical_path_identity_inventory.json"),
        rule="exclude all prior measurement path identities conservatively, including all long-outcome documents; repeat identity-only registry audit before future launch; new historical IDs can only be added, never removed",
        historical_result_values_used_for_selection=False),
    deduplication=dict(against="all 20 historical documents and earlier candidates in frozen order",
        raw_bytes="SHA256(original UTF8 bytes)", normalized_content="SHA256(NFC then whitespace split/join)",
        token_prefix="SHA256(each of first 32769 token IDs packed unsigned32 little-endian)",
        full_tokens="SHA256(all token IDs packed unsigned32 little-endian)",
        keep="earliest eligible candidate in frozen order; historical documents always take precedence",
        normalized_text_is_never_model_input=True),
    ordering=dict(seed=0, rule='ascending SHA256(UTF8(experiment + "|seed=0|" + document_id + "|" + raw_sha256)); document_id lexicographic tie-break',
        policy_sha256=sha(D/"PROSPECTIVE_INPUT_ORDER_POLICY_FREEZE.json"),
        full_input_eligible_order=[x["document"] for x in eligible],
        maximum_calibration_screen_documents=40, target_document_count=4,
        screen_budget_counts="each identity-eligible document with attempted full16 calibration counts once, including uninformative or failed calibration",
        rule_for_extra_identity_exclusions="drop newly historical or duplicate IDs from original order without reordering survivors; no expansion of 40-screen cap"),
    input_semantics=dict(model=spec["model"], tokenizer_encode="encode(original UTF8 text, add_special_tokens=False)",
        token_count=32769, execution_tokens=32768, insufficient_input_behavior="exclude; never pad, concatenate, repeat or truncate horizon"),
    calibration=dict(windows=[256,1024], horizon=128, input_execution_endpoint=1152,
        layers=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22], heads=16,
        rotation_order=roster, rotation_identities=identities["rotations"],
        source_function_SHA256=sources, pair_rule_code_sha256=sha(D/"next_prospective_pair_rule.py"),
        state_error="post-current-QDQ native student state minus matched FP native state from unchanged original recurrence",
        P1="original aggregate.P1: equal mean of readout_energy at starts 256 and 1024, each mean over 18 layers, 16 heads, 128 values; readout=query_readout(E, prepared FP q) from exact historical helpers. Do not substitute P3_H128 or future queries.",
        L7_functional="equal mean across 2 windows, 128 tokens/window, 18 layers of sum((actual student post-o_proj - FP post-o_proj)^2)/max(sum(FP post-o_proj^2),1e-12)",
        L7_state="equal mean across 2 windows, 128 tokens/window, 18 layers, 16 heads of sum_K,V((student state-FP state)^2)/max(sum_K,V(FP state^2),1e-12)",
        L7_total="L7_FUNCTIONAL_ORIGINAL + 0.1*L7_STATE_ORIGINAL; unchanged per-token helper reduction then window means",
        historical_L7_training="H128 recurrent-aware; calibration is matched full-model deployment scoring, not a replay of original FP-conditioned training sampling",
        FP_reference="one fresh same-runtime same-GPU matched short FP reference per document; no score joins across GPUs/references"),
    pair_eligibility=dict(metrics=["P1","L7_TOTAL_ORIGINAL"], lower_is_safer=True,
        opposite_sign="sign(P1[a]-P1[b])*sign(L7_TOTAL[a]-L7_TOTAL[b]) == -1",
        rank="original exact-equality midranks over all 16 rotations, one-based, no fuzzy ties",
        minimum_midrank_gap_each_metric=3,
        priority="descending min(P1 rank gap,L7_TOTAL rank gap), then lexicographic ordered (condition_a,condition_b)",
        uninformative_document="log full16 scores and skip under frozen screening policy before any long outcomes",
        cohort_selection="first four informative documents in frozen order; freeze all four documents, pairs, predictions, input/ref/code/runtime hashes before any 32K Primary outcome",
        fewer_than_four="INSUFFICIENT_DISCRIMINATIVE_DOCUMENTS_STOP; retain all screening evidence and run zero new 32K units; no relaxed gap, no 41st screen, no partial-cohort declaration",
        calibration_gate_failure="stop entire planned extension; preserve first failure, no replacement/retry as a scientific result"),
    long_execution=dict(mode="single-cache serial GPU6", GPU_UUID="GPU-de53f243-b271-521c-d875-d96611ed8574",
        document_order="frozen cohort order", within_document_order=["FP","P1_predicted_rotation","L7_predicted_rotation"],
        planned_FP_units=4, planned_student_units=8, maximum_32K_scientific_units=12,
        each_history="each condition starts token0 in its own coordinate/history; complete own8192 checkpoint only",
        qualification="new scientific keys: source/token/rotation identity, fresh-process own64-token resume parity and observer ON/OFF equality before first Primary reduction; unchanged qualified engine may reuse contract qualification",
        FP_pairing="same document, frozen runtime/GPU/model/tokens/positions; save real float32 full-V logits, SHA and complete native checkpoint, disk/CPU stream to each student",
        persistent_semantics="original every-token INT8-R128 QDQ writeback and next-kernel consumption, original KDA/MLA/conv/positions; teacher-forced",
        runtime=spec["runtime"], engine_sha256=spec["engine_sha256"],
        qualification_failure="first-failure global STOP; no tolerance relaxation; remaining units NOT_RUN"),
    primary=dict(spec["primary"], original_resolution=spec["resolution"],
        signed_delta="KL(P1_predicted_rotation)-KL(L7_TOTAL_predicted_rotation)",
        P1_support="delta < -3.3300064494019496e-6", L7_support="delta > +3.3300064494019496e-6",
        unresolved="absolute delta <= frozen resolution", threshold_update_allowed=False,
        changed_contract="STOP or new separately preregistered experiment; never widen original tolerance after seeing outcomes"),
    secondary=dict(allowed="only frozen V4 state L2/MSE/norms at 8192,16383,24575,32767 and frozen quantization diagnostics",
        E_future="NOT_FROZEN_NOT_COMPUTED", Hidden_KL="NOT_FROZEN_NOT_COMPUTED"),
    missing_and_failure=dict(required_for_valid_document="both complete finite 32768 student runs with exact384 scores and qualified matched FP reference",
        no_imputation=True, no_failed_unit_as_loss=True, no_missing_as_zero=True,
        no_replacement_after_cohort_freeze=True, no_retry_after_numerical_failure=True,
        global_stop="preserve completed partial evidence with statuses; do not claim planned4 complete; no confirmatory analysis with missing documents"),
    analysis=dict(independent_unit="document, not position/window/rotation contrast",
        planned_documents=4, population="eligible unseen PG19 test documents with prespecified P1/L7-discordant natural rotation pair and both rank gaps>=3",
        report="per-document frozen predictions, primary KLs, signed deltas, empirical-resolution labels; counts P1-supported/L7-supported/unresolved/missing with planned denominator4 and valid denominator explicitly separate",
        primary_summary="document-level direction counts; descriptive effect magnitudes, no pooled mean to declare a stable advantage",
        p_values="no confirmatory p-value and no token-level p-value for this small prospective extension",
        all_four_support_P1="small conditional prospective extension, not final statistical confirmation; retain original2-doc MIXED in separate table",
        larger_confirmation="requires separate preregistration/sample-size analysis and explicit authorization; never auto-start",
        stable_advantage_or_gradient_trainability_inference=False),
    authorization=dict(current="protocol and CPU input identity preparation only", future_short_calibration=False,
        future_32K=False, training=False, AIME=False, free_generation=False))
save("DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json", rules)
timing=read(D/"PROSPECTIVE_RESOURCE_TIMING_BASIS.json")
cal=timing["calibration_gpu_hours"]
prior_ledger_path=OLD/"delivery/RESOURCE_LEDGER.json"
prior_ledger=read(prior_ledger_path)
long_base=prior_ledger["measured_completed_worker_hours"]
plan=dict(status="PLAN_ONLY_NOT_EXECUTED", GPU_scientific_units_executed=0, actual_new_document_GPU_hours=0,
    hardware="single available RTX3090 24GB GPU6; same UUID/runtime as current qualified serial pipeline",
    calibration=dict(maximum_screen_documents=40, forward_branches_each_document=17,
        branch_endpoint=1152, includes="1 matched FP+16 frozen rotations; no training",
        historical_same_GPU6_hours_per_full16_document=cal, historical_timing_basis=timing,
        estimated_hours_if_first4_informative=4*cal, estimated_hours_at_max40=40*cal),
    formal=dict(FP_32K_units=4, student_32K_units=8, total_32K_units=12,
        estimate_basis="double prior2-document 6-unit serial trajectory+own-checkpoint qualification worker wall accounting",
        historical_prior2doc_total_worker_hours=long_base, estimated_total_hours=2*long_base,
        historical_ledger_sha256=sha(prior_ledger_path),
        historical_accounting_method=prior_ledger["GPU_hours_method"],
        qualification_and_model_loading_included=True,
        historical_teardown_interval_excluded=True,
        future_actual_accounting="worker launch through natural exit including context teardown; margin covers variation, not a measured teardown estimate"),
    total_estimate=dict(first4_informative=4*cal+2*long_base, maximum40_screen=40*cal+2*long_base,
        planning_margin_multiplier=1.2, maximum40_with_margin=(40*cal+2*long_base)*1.2,
        estimate_is_not_hard_GPUh_cap=True, actual_ledger_required=True),
    physical=dict(no_fixed_85percent_VRAM_gate=True, no_three_cache_residency=True,
        empirical_qualified_32768_peak_allocated_bytes=18087691264,
        empirical_qualified_32768_peak_reserved_bytes=24916262912,
        physical_device_total_bytes=25296044032, native_cache_growth_bytes_per_token=61440,
        FP_streaming="real full-V FP32 CPU/disk logits; no concurrent full GPU FP branch",
        admission="GPU UUID/ownership/runtime/model/rotation/checkpoint/disk checks before every worker; actual measured cache geometry and qualified full32768 evidence",
        disk_start_free_floor_bytes=64*1024**3,
        disk_planning="reserve at least64GiB for own complete checkpoints, FP payloads, parity traces, logs, manifests; recompute file-size projection before launch",
        stop="actualOOM, confirmed nonprogress, illegal native cache growth, failed finite/identity/parity or inability to retain assets",
        nonprogress_timeout_seconds=900, no_global_GPUh_hard_cap=True),
    future_runner_readiness="only input/path/whitelist transport adaptation permitted; freeze a clean future worktree and qualify every affected adapter before authorization/execution; mathematical helpers/quantizer unchanged",
    automatic_launch=False)
save("NEXT_PROSPECTIVE_RESOURCE_PLAN.json",plan)
lines=["# 下一轮四篇独立 PG19 文档：前瞻扩展预注册", "",
    "协议和 CPU 输入身份已准备完成；本轮不执行这些文档的 calibration 或 32K。原两篇正式结论 MIXED 永久保留。", "",
    "候选池为官方完整 PG19 test 的100篇，来源为[PG19官方仓库](https://github.com/google-deepmind/pg19)。官方对象 generation、MD5、原文 SHA256、冻结 tokenizer 编码、完整 token 与32769-token前缀身份均已实际核验。20篇历史文档被保守排除，另22篇不足32769 tokens，58篇输入资格可用。", "",
    "完整逐文档身份、排序与排除理由见 PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT.json；规则见 DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json。CPU输入资格不表示短程分歧配对资格，更不表示已选出四篇。", "",
    "历史排除："+", ".join(excluded)+"。该清单绑定真实服务器历史路径登记；未来启动前只核对身份新增，新增历史文档继续排除，不能恢复旧文档。", "",
    "排序固定为 seed=0，按 SHA256(experiment + '|seed=0|' + document_id + '|' + raw_sha256) 升序，document_id打破相同哈希。历史文档优先占用去重身份；对完整原文、NFC加空白归一化原文、完整tokens和32769前缀tokens作精确哈希去重；归一化文本只用于去重，模型输入保持原文。", "",
    "按冻结顺序最多筛选40篇输入合格文档，每篇先完成16个旋转的同参考短程评分。自然无分歧配对可在看到任何长程outcome前按预定顺序继续筛选；calibration身份、数值或机械失败则停止整轮。选前四篇信息性文档，所有文档和pair必须一起冻结后才启动任何长程Primary。40篇内不足四篇则零32K启动，保存原因，不放宽gap、不追加第41篇、不把部分文档当完整四篇。", "",
    "冻结16旋转："+", ".join(roster)+"。每项的路径、SHA和layer覆盖见规则JSON；已有L8候选仅作为冻结roster的一员评分，不进行任何新L8训练。", "",
    "Calibration连续执行0→1152，windows starts=256/1024、H128、18个KDA层、全部16 heads、两窗等权。每个student从自身token0建立INT8-R128状态；教师为同GPU/runtime的匹配FP模型，所有16项共享本次同一短程参考。", "",
    "P1保留原K3 Query-Readout-aware定义。原实现的aggregate.P1是两个start时点readout_energy的等权均值：post-QDQ native状态误差与原prepared FP query相乘后，对key维求和，再对18层、16 heads、128 value维求平方均值。prepared_query/query_readout及query归一化均直接使用绑定SHA的原helper，不改成P3_H128，也不宣称计算全部未来Query。", "",
    "L7_FUNCTIONAL_ORIGINAL是每token、每层实际post-o_proj误差平方和除以FP输出平方和（clamp_min=1e-12），再对层、128 tokens、两窗等权。L7_STATE_ORIGINAL是每层每head native状态误差平方和除以FP状态平方和（clamp_min=1e-12），再对head、层、token、窗等权。L7_TOTAL_ORIGINAL保留 functional + 0.1*state。原L7训练已是H128 recurrent-aware；这里复用其原loss helper对匹配full-model部署tensor评分，不冒充重放原FP-conditioned训练采样。", "",
    "完整16项低分更安全，严格数值相等才midrank并列。pair要求两指标分数差异符号相反，双方midrank gap≥3。按较大的min(gap_P1,gap_L7)优先，再按排序后的科学key pair字典序打破并列。next_prospective_pair_rule.py保留原三个helper的AST，不加其他偏好。", "",
    "长程顺序：按冻结文档顺序，逐篇FP→P1预测旋转→L7预测旋转；总4 FP+8 student=12条完整32K科学轨迹。逐条件自身0→8192历史与完整checkpoint，blind fresh-process8192→8256恢复及observer ON/OFF必须exact。学生均every-token QDQ、写回、下一kernel消费，保留KDA/MLA/conv/position语义。FP完整32768执行并保存真实FP32 full-V logits及完整身份，CPU/disk streaming单cache串行配对。", "",
    "Primary固定V=157184，KL(FP||student)，t=8192+64*k、k=0..383，same-step logits无shift，Python sum/384，模型实际执行32768。数值分辨率固定3.3300064494019496e-6；Δ=KL(P1预测旋转)-KL(L7预测旋转)，低于负阈值支持P1，高于正阈值支持L7，其余UNRESOLVED。该阈值仅是原合格contract的经验数值分辨率，不是显著性。新key受影响数值资格必须PASS，不能据新outcome改阈值；新contract不兼容则STOP或另立实验。", "",
    "Secondary仅原四时点V4 state L2/MSE/norm及冻结量化诊断。E_future、Hidden KL仍NOT_FROZEN_NOT_COMPUTED。不得新增正式功能proxy。", "",
    "每文档要两student及对应FP全部完整有限、384位置齐全、checkpoint/配对合格才是valid。任何缺失不填零、不按失败算输、不补文档；发生fail-closed后保存已完成证据，后续NOT_RUN。报告计划分母4与实际valid分母，不能声称完整四篇。", "",
    "独立分析单位是document。主摘要是P1支持/L7支持/UNRESOLVED/MISSING的文档计数与逐文档Δ，不对384相关位置做p-value，也不凭合并平均Δ宣布稳定优势。本轮不做confirmatory p-value。四篇全支持P1也只称有筛选条件的小样本Prospective Extension；原两篇MIXED单列，不能与旧文档混成新的成功率。数据去重保障输入不重复，跨文档独立性仍是外推假设。", "",
    "更大独立confirmation需要另行预注册、样本量设计与明确授权；本轮不自动进行。交叉面板不能回答P1梯度可优化性、新L8/L9或reasoning改善。本协议也不授权P1梯度、Cayley optimizer step、AIME或free generation。", "",
    "执行准备：先把现有合格short-calibration adapter的输入路径/CLI白名单移植到新clean worktree，保持上述helper与数学AST，冻结code/runtime/source/旋转SHA并完成受影响GPU门禁。模型与未变历史engine资格可依法复用；此处不把协议和CPU选择函数称为已通过未来GPU执行资格。", "",
    f"资源计划：同GPU6历史16项短程约{cal:.3f} GPUh/文档；四篇最早全合格约{plan['total_estimate']['first4_informative']:.2f} GPUh（含正式轨迹和checkpoint资格），最多40篇筛选约{plan['total_estimate']['maximum40_screen']:.2f} GPUh。估算不设人为GPUh硬上限；实际OOM/无进展/非法cache/IO/数值失败STOP。细项见NEXT_PROSPECTIVE_RESOURCE_PLAN.json。", ""]
(D/"NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md").write_text("\n".join(lines))
print(json.dumps(dict(status=rules["status"], candidate_pool=100,input_eligible=58,selected_documents=0,new_document_GPU_science=0)))
