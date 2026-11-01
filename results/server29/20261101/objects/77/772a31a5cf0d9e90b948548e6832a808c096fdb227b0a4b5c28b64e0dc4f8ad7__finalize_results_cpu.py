"""Seal actual completed exploratory results, or a genuine first-failure report."""
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

D = Path(os.environ["LING_CROSS_RESULTS"])
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def save(name, value):
    p=D/name
    tmp=p.with_name(p.name+".partial.cpu")
    with tmp.open("x",encoding="utf8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()


def csv_file(name, rows):
    assert rows
    with (D/name).open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def markdown(name, lines):
    with (D/name).open("x",encoding="utf8") as f:f.write("\n".join(lines)+"\n")


def direction(delta,resolution,a,b):
    return a if delta < -resolution else b if delta > resolution else "UNRESOLVED"


def main():
    spec=read(D/"FROZEN_SPEC.json");old=Path(spec["reference_experiment"])
    complete=(D/"CROSS_COMPLETE.json").exists();stop=read(D/"STOP.json") if (D/"STOP.json").exists() else None
    assert complete or stop,"WAIT_FOR_TERMINAL_STATUS"
    assert not (complete and stop),"CONTRADICTORY_TERMINAL_STATUS"
    resolution=spec["resolution"]["threshold"]
    asset=read(D/"EXISTING_RESULTS_ASSET_AUDIT.json")
    assert asset["status"]=="PASS" and read(D/"FP_REFERENCE_REUSE_GATE.json")["status"]=="PASS"
    protocol=read(D/"NEXT_PROSPECTIVE_PROTOCOL_FREEZE.json")
    for name,value in protocol["files"].items():assert sha(D/name)==value,("FUTURE_PROTOCOL_CHANGED",name)
    wt=Path(spec["source_worktree"])
    assert subprocess.check_output(["git","-C",str(wt),"status","--porcelain"],text=True)==""
    assert subprocess.check_output(["git","-C",str(wt),"rev-parse","HEAD"],text=True).strip()==spec["source_commit"]
    for filename,expected in read(D/"FORMAL_CODE_FREEZE.json")["files"].items():assert sha(D/filename)==expected
    for name,row in spec["source_assets"].items():
        stat=Path(name).stat()
        assert stat.st_size==row["bytes"] and stat.st_mtime_ns==row["mtime_ns"],("BOUND_ASSET_METADATA_CHANGED",name)
    for name,expected in spec["historical_STOP_sha256"].items():assert sha(name)==expected,("HISTORICAL_RECEIPT_CHANGED",name)
    records={};sources={}
    for row in asset["old_student_records"]:
        p=Path(row["path"]);assert sha(p)==row["sha256"]
        r=read(p);key=(r["document"],r["condition"]);records[key]=r
        sources[key]=dict(path=str(p),sha256=sha(p),origin="REUSED_OLD_COMPLETE",
            source_commit_of_engine=row["source_commit_of_engine"],actual_adapter_code_freeze_sha256=row["actual_successful_adapter_freeze_SHA"])
    new_rows=[];identities=[]
    for unit in spec["new_units"]:
        doc,rot=unit["document"],unit["rotation"];key=(doc,rot)
        p=D/"formal"/f"{doc}__{rot}__formal.json"
        if not p.exists():
            assert not complete
            continue
        r=read(p);assert r["status"]=="PASS" and r["endpoint"]==32768
        assert r["all_finite"] and r["formal_observer_uninterrupted_parity_PASS"]
        assert r["observer_qualified_before_first_primary_reduction"]
        assert r["persistent_integrity"]["passed"]
        assert r["persistent_integrity"]["every_token_layer_qdq_count"]==32768*18
        assert r["persistent_integrity"]["actual_next_kernel_equal_post_qdq_count"]==(32768-1)*18
        assert r["token_sha256"]==unit["token_file"]["sha256"]
        assert r["gpu_uuid"]==spec["execution"]["gpu_uuid"]
        assert r["primary_spec_sha256"]==sha(D/"FROZEN_SPEC.json")
        assert r["vocabulary_dimension"]==157184
        cp=D/"runs"/f"{doc}__{rot}__8192.pt";assert sha(cp)==r["checkpoint_sha256"]
        parent=read(D/"runs"/f"{doc}__{rot}__parent.json")
        resumed=read(D/"runs"/f"{doc}__{rot}__resumed_parity.json")
        for field in ["uninterrupted_suffix_full_logit_sha256","final_full_native_cache_sha256","audit_state_fingerprint","checkpoint_sha256"]:
            assert parent[field]==resumed[field]
        fp_path=old/"formal"/f"{doc}__FP__formal.json";fp=read(fp_path)
        receipt=D/"formal"/f"{doc}__{rot}__formal.NATURAL_EXIT.json"
        assert receipt.exists() and read(receipt)["result_sha256"]==sha(p)
        sources[key]=dict(path=str(p),sha256=sha(p),origin="NEW_CROSS_COMPLETE",source_commit=spec["source_commit"],actual_adapter_code_freeze_sha256=sha(D/"FORMAL_CODE_FREEZE.json"))
        records[key]=r
        allocated=max(x["peak_allocated"] for x in r["memory"]);reserved=max(x["peak_reserved"] for x in r["memory"])
        new_rows.append(dict(document=doc,rotation=rot,classification="POSTHOC_EXPLORATORY_CROSS_PANEL",
            primary_mean_KL=r["primary_mean_KL"],endpoint=32768,score_count=len(r["primary_samples"]),
            GPU_UUID=r["gpu_uuid"],formal_worker_wall_hours_through_natural_exit=read(receipt)["wall_seconds"]/3600,
            formal_peak_allocated_bytes=allocated,formal_peak_reserved_bytes=reserved,
            token_sha256=r["token_sha256"],rotation_sha256=unit["rotation_identity"]["sha256"],
            checkpoint_sha256=r["checkpoint_sha256"],formal_result_sha256=sha(p),source_commit=spec["source_commit"]))
        stages=[]
        for stage in ["parent","resumed_parity","formal"]:
            folder=D/("formal" if stage=="formal" else "runs");tag=f"{doc}__{rot}__{stage}"
            stages.append(dict(stage=stage,result_path=str(folder/(tag+".json")),result_sha256=sha(folder/(tag+".json")),
                natural_exit_receipt=read(folder/(tag+".NATURAL_EXIT.json")),receipt_sha256=sha(folder/(tag+".NATURAL_EXIT.json"))))
        identities.append(dict(document=doc,rotation=rot,token_identity=unit["token_file"],raw_text_identity=unit["raw_text"],
            rotation_identity=unit["rotation_identity"],model_identity=spec["model"],
            model_weights_SHA_records={n:v for n,v in spec["source_assets"].items() if n.endswith(".safetensors")},
            source_commit=spec["source_commit"],source_worktree=str(wt),source_worktree_clean=True,
            numerical_engine_sha256=spec["engine_sha256"],source_adapter_freeze_sha256=sha(D/"FORMAL_CODE_FREEZE.json"),
            numerical_runtime=r["runtime"],GPU_UUID=r["gpu_uuid"],
            reused_FP_manifest=dict(path=str(fp_path),sha256=sha(fp_path),sampled_logits_content_sha256=fp["sampled_logits_content_sha256"],
                original_primary_spec_sha256=fp["primary_spec_sha256"],payload_identity_gate_sha256=sha(D/"FP_REFERENCE_REUSE_GATE.json")),
            own_rotation_checkpoint=dict(path=str(cp),sha256=sha(cp),next_position=8192,parent_full_cache_SHA_at8256=parent["final_full_native_cache_sha256"],resume_exact=True),
            scoring_contract_sha256=spec["scoring_contract_sha256"],sample_positions=spec["primary"]["positions"],
            reducer="original GPU FP32 full-vocab KL(FP||student), Python sum/384",stage_receipts=stages,
            complete_run_finite=True,complete_endpoint=32768,peak_allocated=allocated,peak_reserved=reserved))
    for key,r in records.items():
        assert r["status"]=="PASS" and r["endpoint"]==32768 and r["all_finite"]
        assert r["sample_positions"]==spec["primary"]["positions"]
        assert [x["position"] for x in r["primary_samples"]]==spec["primary"]["positions"]
        assert all(x["vocabulary_dimension"]==157184 and math.isfinite(x["KL"]) for x in r["primary_samples"])
        assert sum(x["KL"] for x in r["primary_samples"])/384==r["primary_mean_KL"]
    resources=[];peak_allocated=peak_reserved=0
    for folder in [D/"runs",D/"formal"]:
        for p in sorted(folder.glob("*.NATURAL_EXIT.json")):
            receipt=read(p);result_path=p.with_name(p.name.replace(".NATURAL_EXIT.json",".json"));result=read(result_path)
            assert receipt["natural_exit"] and receipt["result_sha256"]==sha(result_path)
            memory=result.get("memory",[])
            allocated=max(x["peak_allocated"] for x in memory);reserved=max(x["peak_reserved"] for x in memory)
            peak_allocated=max(peak_allocated,allocated);peak_reserved=max(peak_reserved,reserved)
            resources.append(dict(tag=receipt["tag"],worker_wall_seconds=receipt["wall_seconds"],
                worker_wall_hours=receipt["wall_seconds"]/3600,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                natural_exit_receipt_sha256=sha(p),result_sha256=sha(result_path),source_commit=receipt["source_commit"]))
    actual_hours=sum(x["worker_wall_hours"] for x in resources)
    nvml_sample_peak_mib=0;nvml_sample_count=0
    for folder in [D/"runs",D/"formal"]:
        for p in folder.glob("*.GPU_EXCLUSIVITY.jsonl"):
            for line in p.read_text().splitlines():
                guard=json.loads(line)
                assert guard["exclusive"] and not guard["failure_conditions"]
                for line in guard["compute"]["stdout"].splitlines():
                    fields=[x.strip() for x in line.split(",")]
                    if fields[0]==spec["execution"]["gpu_uuid"]:
                        nvml_sample_peak_mib=max(nvml_sample_peak_mib,float(fields[-1]));nvml_sample_count+=1
    incomplete=[]
    if stop:
        for folder in [D/"runs",D/"formal"]:
            for p in folder.glob("*.STARTED.json"):
                receipt=p.with_name(p.name.replace(".STARTED.json",".NATURAL_EXIT.json"))
                if receipt.exists():continue
                j=read(p);begin=j.get("started",j.get("started_epoch"));end=stop.get("epoch",stop.get("stopped_epoch",time.time()))
                incomplete.append(dict(tag=p.stem,started_epoch=begin,STOP_epoch=end,
                    failed_worker_hours_upper_bound_to_STOP=max(0,end-begin)/3600,teardown_not_observed_in_natural_exit_receipt=True))
    save("RESOURCE_USAGE_LEDGER.json",dict(status="COMPLETE" if complete else "STOP",GPU_UUID=spec["execution"]["gpu_uuid"],
        hardware="single RTX3090 24GB GPU6",GPU_hours_method="sum serial GPU-worker launch-to-natural-exit wall hours, including startup/model loading and context teardown; CPU-only audits/plotting/protocol and idle controller waits excluded; not utilization-weighted CUDA kernel time",
        completed_worker_hours=actual_hours,failed_worker_hours_to_STOP=incomplete,
        completed_worker_count=len(resources),workers=resources,
        peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
        CUDA_peak_scope="recorded qualified token-execution phases; original workers reset peak statistics at loop start",
        NVML_process_memory_sample_peak_MiB=nvml_sample_peak_mib,NVML_process_memory_sample_count=nvml_sample_count,
        NVML_scope="exact physical UUID compute-process memory samples from periodic ownership guards; a sampled peak, not continuous high-water tracking",
        disk_free_bytes=shutil.disk_usage(D).free,scientific_new_students_complete=len(new_rows),
        new_formal_students_started=len(list((D/"formal").glob("*__formal.STARTED.json"))),
        own_history_qualification_workers=len(list((D/"runs").glob("*.STARTED.json"))),
        FP_reruns=0,old_student_reruns=0,new_document_GPU_science=0,training_steps=0,
        no_GPUh_hard_cap=True,no_85percent_memory_gate=True,single_cache_serial=True,STOP=stop))
    save("NEW_SCIENTIFIC_UNIT_IDENTITIES.json",dict(status="COMPLETE" if complete else "PARTIAL_OR_NOT_RUN",units=identities))
    analysis=[];matrix=[];contrasts=[]
    rotations=["H_to_L6_a50","H_to_L7_a25","H_to_L7_a50"]
    if complete:
        assert len(new_rows)==2 and len(resources)==6 and len(records)==6
        qual=read(D/"CROSS_RUN_NUMERICAL_QUALIFICATION.json");assert qual["status"]=="PASS"
        for item in qual["resources"]:
            assert records[(item["document"],item["rotation"])]["final_cache_inventory"]["cache_bytes"]==item["predicted_native_cache_32768"],"ILLEGAL_COMPLETE_NATIVE_CACHE_GROWTH"
        for doc in ["pg19-test-860","pg19-test-5956"]:
            row=dict(document=doc,**{r:records[(doc,r)]["primary_mean_KL"] for r in rotations},
                status="COMPLETE_ALL3_ROTATIONS",scored_positions=384,endpoint=32768,
                classification="POSTHOC_EXPLORATORY_CROSS_PANEL")
            matrix.append(row)
            for name,a,b in [("A",rotations[0],rotations[1]),("B",rotations[0],rotations[2]),("C",rotations[1],rotations[2])]:
                delta=row[a]-row[b]
                contrasts.append(dict(document=doc,contrast=name,rotation_a=a,rotation_b=b,KL_a=row[a],KL_b=row[b],
                    delta_a_minus_b=delta,frozen_384_primary_resolution=resolution,
                    empirical_resolution_label=direction(delta,resolution,a,b),inferential_status="POSTHOC_EXPLORATORY_NOT_CONFIRMATORY"))
        csv_file("CROSS_ROTATION_32K_RESULTS.csv",new_rows)
        csv_file("TWO_DOC_THREE_ROTATION_MATRIX.csv",matrix)
        csv_file("CROSS_PANEL_ROTATION_CONTRASTS.csv",contrasts)
        analysis=["# 两文档 × 三旋转：探索性比较","", "原两篇正式预测胜负保持860=P1、5956=L7，Overall=MIXED（1/2），本面板不重选旧pair或重算原成功率。", "",
            "|文档|H_to_L6_a50|H_to_L7_a25|H_to_L7_a50|", "|---|---:|---:|---:|"]
        analysis += [f"|{x['document']}|{x[rotations[0]]:.12g}|{x[rotations[1]]:.12g}|{x[rotations[2]]:.12g}|" for x in matrix]
        analysis += ["","|文档|contrast|Δ(a−b)|原384点数值分辨率下更低的旋转|","|---|---|---:|---|"]
        analysis += [f"|{x['document']}|{x['rotation_a']} vs {x['rotation_b']}|{x['delta_a_minus_b']:.12g}|{x['empirical_resolution_label']}|" for x in contrasts]
        interaction=False;unresolved=False
        for name in ["A","B","C"]:
            rows=[x for x in contrasts if x["contrast"]==name];labels=[x["empirical_resolution_label"] for x in rows]
            if "UNRESOLVED" in labels:
                sentence="至少一篇处于经验数值未分辨范围，不能认定相同或反转排序。";unresolved=True
            elif labels[0]==labels[1]:sentence="两篇给出相同排序；仅是两篇上的一致观察，不能宣称稳定旋转效应。"
            else:sentence="两篇给出反转排序，构成这两个文档上的探索性 document×rotation 交互信号。";interaction=True
            analysis += ["",f"Contrast {name}：860 更低={labels[0]}；5956 更低={labels[1]}。{sentence}"]
        p1_patterns=[]
        for doc in ["pg19-test-860","pg19-test-5956"]:
            rows=[x for x in contrasts if x["document"]==doc and x["contrast"] in ["A","B"]]
            p1_patterns.append(f"{doc}：相对a25={rows[0]['empirical_resolution_label']}，相对a50={rows[1]['empirical_resolution_label']}")
        analysis += ["","原P1选择的L6-a50对两个L7候选："+"；".join(p1_patterns)+"。"]
        if interaction:
            explanation="同一旋转contrast在两文档间发生排序反转，说明原MIXED不能只用两篇选了不同L7候选来概括；文档条件与旋转相对表现共同构成探索性解释。具体因果路径仍无法识别。"
        elif unresolved:
            explanation="面板包含数值未分辨contrast，候选差异与文档条件的贡献不能完全拆解。原MIXED仍保留。"
        else:
            explanation="三个相同旋转contrast在两篇中排序均一致；原pair使用不同L7候选可作为原MIXED的候选选择解释线索。但两篇不足以确认稳定候选效应，也不能据此改选旧L7候选。"
        analysis += ["",explanation,"", "全部对比仍使用原384点Primary与冻结经验分辨率。六个contrast不是六个独立文档；没有位置级p-value、交互显著性检验或总体稳定效应结论。后续独立四篇是另一个预注册扩展，本轮不执行。"]
        markdown("DOCUMENT_ROTATION_INTERACTION_ANALYSIS.md",analysis)
        interaction_status="EXPLORATORY_ORDER_REVERSAL_SIGNAL_TWO_DOCUMENTS" if interaction else "NOT_IDENTIFIABLE_FOR_UNRESOLVED_CONTRASTS" if unresolved else "NO_ORDER_REVERSAL_OBSERVED_TWO_DOCUMENTS"
        save("COMPLETED_CROSS_PANEL_AUDIT.json",dict(status="PASS",new_units_complete=2,old_units_reused=4,FP_reused=2,
            all6_exact384_reducers_verified=True,new2_complete_run_finite_PASS=True,
            new2_own_resume_and_observer_parity_PASS=True,unchanged_frozen_source_worktree_clean=True,
            source_commit=spec["source_commit"],original_formal_verdict="MIXED_PERMANENTLY_PRESERVED",
            matrix_sources=[dict(document=k[0],rotation=k[1],**v) for k,v in sources.items()],
            original_qualification_sha256=sha(D/"CROSS_RUN_NUMERICAL_QUALIFICATION.json"),
            resource_ledger_sha256=sha(D/"RESOURCE_USAGE_LEDGER.json")))
    else:
        interaction_status="NOT_IDENTIFIABLE_CROSS_PANEL_INCOMPLETE"
        explanation="交叉面板未完整完成，不能识别三旋转排序或文档×旋转交互；保留原MIXED，交付真实STOP和下一轮协议。"
        if not (D/"CROSS_RUN_NUMERICAL_QUALIFICATION.json").exists():
            save("CROSS_RUN_NUMERICAL_QUALIFICATION.json",dict(status="FAIL_OR_INCOMPLETE_FIRST_FAILURE_STOP",STOP=stop,
                completed_qualification_worker_receipts=[x for x in resources if "__formal" not in x["tag"]],
                no_CPU_test_substituted_for_GPU_gate=True,no_tolerance_relaxation=True))
    statuses=dict(EXISTING_DATA_DIAGNOSIS_STATUS="COMPLETE_SAVED384_SCALARS_AVAILABLE",
        TEMPORAL_FUNCTIONAL_DIAGNOSIS_STATUS="COMPLETE_CPU_POSTHOC_DESCRIPTIVE_CAUSAL_PATH_NOT_IDENTIFIABLE",
        CROSS_ROTATION_PANEL_STATUS="COMPLETE_POSTHOC_EXPLORATORY_CROSS_PANEL" if complete else "STOP_INCOMPLETE_NOT_EVALUATED",
        DOCUMENT_ROTATION_INTERACTION_STATUS=interaction_status,
        NEXT_PROSPECTIVE_READINESS="PREREGISTRATION_FULL100_INPUT_IDENTITIES_AND_RULES_READY_NOT_AUTHORIZED_TO_RUN",
        OVERALL_SCIENTIFIC_VERDICT="ORIGINAL_FORMAL_MIXED_PRESERVED_NO_STABLE_P1_ADVANTAGE_CONFIRMED")
    save("FINAL_STATUS.json",statuses)
    facts=read(D/"TEMPORAL_MECHANISM_FACTS.json");blocks={x["document"]:x for x in facts["documents"]}
    lines=["# Ling/KDA 原 MIXED 的时序诊断与最小交叉面板","",
        "原正式P1-vs-L7结果永久保持：860支持P1，5956支持L7，Overall=MIXED、1/2。新增面板为事后探索，不是独立前瞻确认。", "",
        "1. 原384点是否保存逐位置数值？是。旧4条student的实际primary_samples各384个有限KL标量，精确positions、文档、旋转、V=157184完整；重新Python sum/384与原均值逐项一致。不是从aggregate或SHA反推曲线。", "",
        "2. 860的P1优势集中在哪段？Middle贡献最大。Early meanΔ=-0.0125111464503、Middle=-0.0434400863617、Late=+0.00306729909320；对原384点总Δ的贡献分别-0.00417038215011、-0.0144800287872、+0.00102243303107。", "",
        "3. 5956的L7优势集中在哪段？Late贡献最大。Early meanΔ=+0.00396383643053、Middle=-0.00680282253450、Late=+0.0133601095695；总Δ贡献分别+0.00132127881018、-0.00226760751150、+0.00445336985649。最大正差值在25600，Δ=+1.17150065303；两篇各自最大的10个正差值均有6个落在Late。逐位置和大差值分布文件保留全部依据，峰值不替代原Primary。", "",
        "4. State MSE和时间KL排序一致吗？并非总一致。两篇24575/32767时点均P1 state MSE更低，但Late平均KL均L7更低。860的Middle方向与相邻P1 MSE更低相符；5956的四个快照则无法代表区间MSE。对应表明确最近真实KL时点及时间距离，不插值。冻结实现先产生same-step logits，再做当前QDQ写回供后续kernel消费；post-QDQ state快照与same-step输出也是不同观测。上述定义差异有助于说明不能互相冒充，不能由四个快照认定具体因果。", "",
        "5. 支持早期损伤主导吗？描述上不支持：最大绝对分段贡献分别Middle/Late。支持的有限观察是两篇分段KL符号反转、后期MSE与区间KL排序不一致、不同文档差异集中在不同段。H1的因果早期损伤主导仍NOT_IDENTIFIABLE，H2/H3/H4仅描述性支持。128点分段不套用原384点分辨率。", "",
        f"6. 补齐两条trajectory了吗？{'是，新增860×L7-a50、5956×L7-a25均完整32768，384评分齐全、finite/INT8写回/next-kernel消费/observer/resume门禁PASS。' if complete else '未补齐，新增完整student='+str(len(new_rows))+'；STOP见STOP.json。未生成假空交叉结果表。'}", "",
        "7. 原FP能合法复用吗？能。实际387个payload/文档（384个真实FP32 full-V logit位置及原state快照），token/position/shape、每payload SHA、logit内容摘要、model/tokenizer/source/runtime、GPU6 UUID及原KL reducer匹配PASS。新增FP重跑0、旧student重跑0。旧V4三缓存STOP保留，成功资产来自最新COMPLETE串行pipeline。", "",
        "8. 完整2×3 KL矩阵：", ""]
    if complete:
        lines += analysis[4:8]
    else:lines += ["未形成合法完整矩阵，NOT_RUN/INCOMPLETE，不能拼入未完成值。"]
    lines += ["", "9. 探索性交互信号："+interaction_status+"。"+("具体六个contrast见DOCUMENT_ROTATION_INTERACTION_ANALYSIS.md。" if complete else "完整面板缺失。"),"",
        "10. 能解释原MIXED吗？"+explanation+" 时间分析另外显示两文档的主贡献段不同、优势沿时间反转。上述观察拆解原结果，但不构成机制因果识别。", "",
        "11. 哪些解释无法识别？不能由四个state快照确定区间损伤、误差方向/Query敏感性/历史累积的因果路径；不能把两个文档的探索面板外推为稳定文档或旋转效应。E_future、Hidden KL未冻结、未计算。P1是局部K3 readout-aware，未直接算全部未来Query；原L7已H128 recurrent-aware。", "",
        "12. 下一轮四篇规则：完整官方100篇输入身份已冻，20历史排除、22长度不足、58输入可用；内容及完整/前缀token去重、seed0哈希排序，最多40篇full16 calibration，windows256/1024 H128全部18层16heads、两窗等权，原P1和L7_TOTAL相反方向且双方midrank gap≥3，最大min-gap后lex pair。前四篇信息性文档及pair在全部长程outcome前共同冻结；不足四篇零32K启动，不放宽、不补超额。完整源SHA与规则见下一轮预注册。", "",
        f"13. 本轮实际GPU科学单元：完整新增student={len(new_rows)}，正式student已启动={len(list((D/'formal').glob('*__formal.STARTED.json')))}；旧4 student和2 FP只复用。另有{len(list((D/'runs').glob('*.STARTED.json')))}个自身历史/独立恢复资格worker，不算额外独立科学unit。下一轮新文档GPU、训练、AIME、free generation均0。", "",
        f"14. 实际资源：已自然退出worker合计{actual_hours:.9f} GPUh（worker launch→natural exit墙钟、含模型加载及GPU上下文退出，非利用率加权kernel时间）；记录的CUDA执行段allocated峰值{peak_allocated} bytes={peak_allocated/1024**3:.3f} GiB，reserved峰值{peak_reserved} bytes={peak_reserved/1024**3:.3f} GiB；NVML进程显存采样峰值{nvml_sample_peak_mib:.0f} MiB。CUDA使用原loop开始reset后的peak统计，NVML为周期采样，并非连续瞬时高水位。GPU6单RTX3090 24GB串行单cache。"+("全部6 worker均自然退出。" if complete else "失败worker到STOP的上界单列ledger，不能当完整轨迹。"),"",
        "15. 有新证据支持P1稳定优势吗？没有可确认的稳定优势证据。原独立两文档仍1胜1负，新的两条仅补事后旋转contrast，没有新增独立文档。不能忽略5956、重选L7候选、合并平均Δ宣布总体P1优势或把384个位置当384样本。", "",
        "16. 应启动P1 gradient trainability吗？本轮证据不足以据此推进或自动授权。这里没有梯度、优化或新训练实验；梯度可优化性与功能预测排序是不同问题。下一步先按已写好的独立前瞻扩展协议另行授权执行；即使四篇全支持P1，也保留小样本和选择条件限制。", "",
        "原冻结AIME26背景仍FP44/60、Hadamard35/60、L6 36/60、L7 33/60；本轮未运行AIME。这些背景不作为交叉面板结论。", "",
        "状态：", ""]
    lines += [f"- {k} = {v}" for k,v in statuses.items()]
    lines += ["", "本轮完成后停止。未自动执行下一轮四篇、L8/L9新训练、P1梯度优化、Cayley optimizer、AIME或free generation。"]
    if stop:lines += ["", "实际first failure："+json.dumps(stop,ensure_ascii=False)]
    markdown("FINAL_SCIENTIFIC_REPORT.md",lines)
    required=["EXISTING_RESULTS_ASSET_AUDIT.json","PER_POSITION_KL_AVAILABILITY.json","TEMPORAL_KL_DIAGNOSIS.csv",
        "STATE_MSE_VS_KL_DIAGNOSIS.md","CROSS_ROTATION_FROZEN_SPEC.json","FP_REFERENCE_REUSE_GATE.json",
        "CROSS_RUN_NUMERICAL_QUALIFICATION.json","RESOURCE_USAGE_LEDGER.json","CROSS_ROTATION_32K_RESULTS.csv",
        "TWO_DOC_THREE_ROTATION_MATRIX.csv","DOCUMENT_ROTATION_INTERACTION_ANALYSIS.md","NEXT_PROSPECTIVE_4DOC_PREREGISTRATION.md",
        "DOCUMENT_SELECTION_AND_EXCLUSION_RULES.json","NEXT_PROSPECTIVE_RESOURCE_PLAN.json","FINAL_SCIENTIFIC_REPORT.md"]
    files={}
    for p in D.rglob("*"):
        rel=p.relative_to(D)
        if not p.is_file() or any(x in rel.parts for x in ["prospective_inputs","__pycache__"]):continue
        if p.suffix==".pt" or ".partial" in p.name or p.name in ["SHA256SUMS.txt","RESULT_AND_DELIVERY_HASHES.json","delivery_wait.log"]:continue
        files[str(rel)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    save("RESULT_AND_DELIVERY_HASHES.json",dict(experiment=spec["experiment"],status="COMPLETE" if complete else "STOP",
        source_commit=spec["source_commit"],original_verdict="MIXED_PERMANENTLY_PRESERVED",files=files,
        required_artifacts={n:("COMPLETE" if (D/n).exists() else "NOT_RUN") for n in required},
        large_remote_payload_bindings="own .pt checkpoints bound in NEW_SCIENTIFIC_UNIT_IDENTITIES; old FP real payloads bound in FP_REFERENCE_REUSE_GATE; all100 raw/token inputs bound in PROSPECTIVE_FULL_POOL_INPUT_IDENTITY_AUDIT",
        self_reference_exclusions=["SHA256SUMS.txt","RESULT_AND_DELIVERY_HASHES.json"],
        live_wait_log_excluded=True))
    checks={name:item["sha256"] for name,item in files.items()}
    checks["RESULT_AND_DELIVERY_HASHES.json"]=sha(D/"RESULT_AND_DELIVERY_HASHES.json")
    with (D/"SHA256SUMS.txt").open("x") as f:
        f.write("".join(f"{value}  {name}\n" for name,value in sorted(checks.items())))
    print(json.dumps(dict(status="COMPLETE_CPU_DELIVERY_SEALED" if complete else "STOP_CPU_DELIVERY_SEALED",
        actual_completed_GPU_worker_hours=actual_hours,new_students_complete=len(new_rows),hash_files=len(checks))))


if __name__=="__main__":main()
