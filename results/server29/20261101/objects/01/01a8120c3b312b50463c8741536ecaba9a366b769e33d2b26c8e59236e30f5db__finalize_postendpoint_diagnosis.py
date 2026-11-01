"""CPU-only independent QA and report closure for an existing-artifact diagnosis."""
import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

TASK = 'QWEN_C6R_C7_CANONICAL60_POSTENDPOINT_DIAGNOSIS_V1'
NA = 'NOT_AVAILABLE_FROM_EXISTING_ARTIFACTS'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def check(value, message):
    if not value:
        raise RuntimeError(message)

def b(value):
    check(value in ('True', 'False'), 'unexpected CSV Boolean')
    return value == 'True'

def emit(root, name, text):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)

def num(x):
    return 'NA' if x is None else f'{x:,.2f}'

def table(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '| ' + ' | '.join(['---'] * len(head)) + ' |'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--builder', required=True)
    a = ap.parse_args()
    d = Path(a.output).resolve()
    check(d.name == TASK and d.exists() and not (d/'SHA256SUMS').exists(), 'unexpected or already frozen output')
    s = read(d/'analysis/diagnosis_summary.json')
    v = s['verdict']; dist = s['distributions']; ab = s['abstain']; cluster = s['cluster']
    inventory = read(d/'provenance/input_artifact_inventory.json')
    src = Path(inventory['endpoint_root'])
    reval = read(d/'provenance/read_only_revalidation.json')
    check(src.parent == d.parent and src != d, 'output/source separation')
    with (d/'analysis/c6r_c7_paired_endpoint_table.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 60 and len({r['sample_id'] for r in rows}) == 60, 'paired rows')
    scores = {arm: {(r['problem_id'], int(r['seed'])): r for r in read(src/f'scores/{arm.lower()}_frozen_v4.json')['records']} for arm in ('C6R','C7')}
    grouped = collections.defaultdict(list)
    for r in rows:
        k = r['problem_id'], int(r['seed'])
        for arm in ('C6R', 'C7'):
            sc = scores[arm][k]
            check(b(r[arm+'_correct']) == sc['v4_correct'], 'independent correctness audit')
            check(b(r[arm+'_abstain']) == (sc['v4_status'] == 'ABSTAIN'), 'independent abstain audit')
            check(int(r[arm+'_generation_tokens']) == sc['generated_token_count'], 'length audit')
            check(b(r[arm+'_EOS']) == sc['eos_seen'], 'EOS audit')
            check(b(r[arm+'_length_hit']) == (sc['finish_reason'] == 'length'), 'ceiling audit')
            check(int(r[arm+'_oscillation_proxy']) == sum(int(r[arm+'_'+m]) for m in ['answer_changes','committed_aba_count','retractions']), 'metric sum')
            for n in (4,8,16):
                q = float(r[f'{arm}_repeated_{n}gram_fraction']); u = float(r[f'{arm}_unique_{n}gram_ratio'])
                check(math.isfinite(q) and 0 <= q <= 1 and abs(q+u-1) < 1e-12, 'ngram metric')
            for name in ['first_commit_token','post_commit_token_count','final_retraction_present','semantic_answer_flip_present']:
                check(r[arm+'_'+name] == NA, 'missingness must not be imputed')
        ca, cb = b(r['C6R_correct']), b(r['C7_correct'])
        category = 'A' if cb and not ca else 'B' if ca and not cb else 'C' if ca else 'D'
        check(category == r['pair_category'], 'independent pair taxonomy')
        check(int(r['delta_generation_length']) == int(r['C7_generation_tokens'])-int(r['C6R_generation_tokens']), 'paired delta')
        grouped[category].append(r)
    check([len(grouped[c]) for c in 'ABCD'] == [3,10,30,17], 'primary count reproduction')
    check(sum(b(r['C7_abstain']) for r in grouped['B']) == 3, 'regression abstain')
    check(all(b(r['C7_EOS']) and b(r['C6R_EOS']) and not b(r['C7_length_hit']) and not b(r['C6R_length_hit']) for c in 'AB' for r in grouped[c]), 'discordant termination')
    new_ab = {r['sample_id'] for r in rows if b(r['C7_abstain']) and not b(r['C6R_abstain'])}
    lost_ab = {r['sample_id'] for r in rows if b(r['C6R_abstain']) and not b(r['C7_abstain'])}
    check(new_ab == set(ab['C7_unique_abstain_ids']) and len(new_ab) == 7 and len(lost_ab) == 2, 'abstain flow')
    qmap = collections.defaultdict(list)
    for r in rows: qmap[r['problem_id']].append(r)
    check(len(qmap) == 30 and all(sorted(int(r['seed']) for r in rs) == [1,2] for rs in qmap.values()), '30x2 QA')
    stable = sorted(q for q, rs in qmap.items() if all(r['pair_category'] == 'B' for r in rs))
    check(stable == ['aime26_11','aime26_27'] and stable == cluster['stable_regression_ids'], 'question clustering')
    panel = read(d/'proposals/discordant_replay_panel.json')
    check(panel['status'] == 'PROPOSAL_ONLY_NOT_RUN' and len(panel['rows']) == 25, 'proposal must not run')
    check(dict(collections.Counter(r['pair_category'] for r in panel['rows'])) == dict(A=3,B=10,C=6,D=6), 'panel counts')
    for c in 'CD':
        check(sorted(r['sample_id'] for r in panel['rows'] if r['pair_category'] == c) == sorted(r['sample_id'] for r in grouped[c])[:6], 'deterministic controls')
    check(all(r['reference_condition'] == ('C7' if r['pair_category'] == 'A' else 'C6R') for r in panel['rows']), 'teacher reference')
    for p, digest in reval['consumed_source_hashes'].items(): check(sha(p) == digest, 'source changed '+p)
    protected = []
    for line in (src/'manifests/final_artifact_sha256.txt').read_text().splitlines():
        digest, name = line.split(None,1); p = Path(name.strip()); p = p if p.is_absolute() else src/p
        check(sha(p) == digest, 'endpoint manifest changed '+str(p)); protected.append(p)
    check(len(protected) == reval['endpoint_final_manifest_files_checked'], 'source manifest count')
    check(all(v[k] == 'PASS' for k in ['INPUT_ARTIFACT_IDENTITY_GATE','PAIRED_SAMPLE_ALIGNMENT_GATE','FROZEN_V4_IDENTITY_GATE','PRIMARY_PAIR_COUNT_REPRODUCTION_GATE']), 'hard gates')

    report = ['# QWEN_C6R_C7_CANONICAL60_POSTENDPOINT_DIAGNOSIS_V1', '', '```text']
    report += [f'{k} = {value}' for k, value in v.items()]
    report += ['```', '', '**顶部计数的限定**：`WITH_OSCILLATION=9` 仅指冻结的文本 oscillation proxy 非零；`WITH_ANSWER_FLIP=2` 仅指相邻已承诺整数 claim 变化。它们不是9条已证实的推理振荡或2条已证实的语义翻转。`TERMINATION_FAILURE_SIGNAL=PRESENT` 指弃答信号；长度上限/缺失 EOS 信号不被支持。', '',
    '## 1. 结论与证据边界', '',
    'C6R 40/60（66.67%），C7 33/60（55.00%），C7 净少7条，点估计差 −11.67个百分点。冻结的 primary paired exact p=0.09228515625（复用原结果，不重新检验）。**没有观察到 C7 优越性，也不能写成“统计证明 C7 更差”。** 这是30道题×2个 seed，而不是60道独立题。', '',
    'C7 退步没有统一的已证实机制：3/10是弃答，7/10是错误答案；全部正常 EOS、无长度上限。存在两道双 seed 稳定退步，也有六道仅一个 seed 退步。重复度的配对差异支持有限的 trajectory/termination 后续诊断假设；全局 oscillation proxy 和答案变化并未一致恶化。不能把这些关联解释成 Final-Hidden objective 的因果效果。', '',
    '本任务只读取既有 raw、Frozen V4 scores、冻结 generation-behavior CSV 和历史结果。没有调用 scorer，没有模型 forward、GPU 测量、token生成、训练、checkpoint选择或rotation导出。所有新分析是 CPU-only、posthoc descriptive；不新增显著性检验或阈值。', '',
    '## 2. 来源与只读闭环', '',
    f'- Canonical60 root：`{src}`',
    f'- Endpoint repository：`{inventory["repository_root"]}`；commit：`{inventory["endpoint_commit"]}`。',
    '- C6R/C7 各60条，严格按 problem_id + seed 对齐；prompt、gold、payload SHA、token count、EOS/finish reason、rotation一致性全部通过。',
    f'- 原 endpoint final manifest {len(protected)} 个文件以及实际消费的外部输入，在分析前后和最终QA中均校验 SHA256，不改源文件。',
    f'- Frozen V4：`{inventory["scorer_identity"]["scorer_path"]}`；SHA256 `{inventory["scorer_identity"]["scorer_sha256"]}`。', '',
    table(['Condition','Rotation SHA256','Theta checkpoint identity'], [[arm,inventory['rotations'][arm]['rotation_sha256'],inventory['rotations'][arm]['theta_checkpoint_sha256']] for arm in ('C6R','C7')]), '',
    '逐 raw path/hash、scored-result hash、部署 manifest、model identity、sample IDs、seed集合见 `provenance/input_artifact_inventory.json`；完整消费输入的 hash 清单见 `provenance/read_only_revalidation.json`。Checkpoint identity来自已冻结 export provenance及每条raw audit，不加载checkpoint或重新导出。', '',
    '## 3. 完整配对分类与13条 discordants', '',
    table(['类别','定义','数量'], [['A','C7 rescue C6R',3],['B','C7 regression C6R',10],['C','both correct',30],['D','both wrong',17]]), '',
    '**Rescue IDs**：'+', '.join(s['taxonomy']['sample_ids']['A'])+'。', '',
    '**Regression IDs**：'+', '.join(s['taxonomy']['sample_ids']['B'])+'。', '',
    '下表 Δ均为 C7−C6R；O 是原 oscillation proxy，F 是相邻已承诺整数 claim 变化次数；它们不代表已确认语义失败。重复度 Δ8gram 以百分点表示。全部13条两侧 EOS=true、length-hit=false。完整问题原文、两侧值、NA字段及指标在 paired CSV 和逐条 summary JSON/CSV。', '']
    report += [table(['Sample','类别','C7弃答','长度 C6R→C7','Δtokens','O C6R→C7','F C6R→C7','Δ8gram pp'], [
        [r['sample_id'],r['pair_category'],r['C7_abstain'],f'{r["C6R_generation_tokens"]}→{r["C7_generation_tokens"]}',r['delta_generation_length'],f'{r["C6R_oscillation_proxy"]}→{r["C7_oscillation_proxy"]}',f'{r["C6R_answer_changes"]}→{r["C7_answer_changes"]}',f'{float(r["delta_repeated_8gram_fraction"])*100:+.2f}']
        for c in 'BA' for r in grouped[c]]), '',
    '## 4. 弃答流量：净增加5，不是新增5', '',
    'C6R弃答4，C7弃答9；共享2，C7独有7，C6R独有2，7−2=净增5。C7独有7条中，3条是regression、4条是both-wrong。C6R独有2条在C7中虽不再弃答，但仍错误，不能算rescue。所有9条C7弃答的Frozen V4理由是 `NO_COMMITTED_FINAL_CLAIM`。', '',
    '- C7独有：'+', '.join(ab['C7_unique_abstain_ids'])+'。',
    '- C6R独有：'+', '.join(ab['C6R_unique_abstain_ids'])+'。',
    '- 共同弃答：'+', '.join(ab['shared_abstain_ids'])+'。', '',
    '三条 correct→abstain 是 aime26_02:seed1、aime26_09:seed2、aime26_11:seed1。新增弃答分布在7道题、两个seed，不是单一problem/seed。其余4条新增弃答没有新增regression，因为C6R原本也错误。故弃答可以直接描述30%的退步，不是多数退步的解释。', '',
    '## 5. 长度、EOS与配对分布', '',
    table(['Condition','mean tokens','median','p25','p75','max','length-hit','EOS','abstain'], [
        [arm]+[num(dist['ALL']['conditions'][arm]['generation_tokens'][k]) for k in ['mean','median','p25','p75','max']]+[dist['ALL']['observed_counts'][arm][k] for k in ['length_hit','EOS','abstain']] for arm in ('C6R','C7')]), '',
    table(['类别','n','mean Δtokens','median','p25','p75','min','max','C7更长'], [[c,dist[c]['samples']]+[num(dist[c]['paired_deltas']['generation_tokens'][k]) for k in ['mean','median','p25','p75','min','max']]+[dist[c]['C7_higher_than_C6R']['generation_tokens']] for c in ['ALL','A','B','C','D']]), '',
    '分位数使用排序后线性插值位置 (n−1)q，只用于分布汇总，不是新增科学指标。退步组5/10变长、5/10变短，长度不支持一致的“过早结束”或“推理过长”解释。C6R唯一长度上限在both-wrong组；C7没有长度上限。EOS出现不等于答案正确，也不等于已形成正式最终claim。', '',
    '## 6. 冻结文本代理：振荡、变化、撤回和重复', '',
    '`oscillation_proxy = answer_changes + committed_aba_count + retractions`。其中retractions包括旧V4事件枚举器识别的所有撤回事件（可能含中间/被拒绝claim），不是“最终答案被撤回”。answer_changes只统计已承诺且未拒绝的整数claim相邻变化，不测semantic reasoning flip。保留旧定义，不增n、window、阈值。', '',
    table(['类别','n','非零 O C6R→C7','claim变化 C6R→C7','撤回事件存在 C6R→C7','mean ΔO','median ΔO','C7 O更高'], [[c,dist[c]['samples']]+[f'{dist[c]["observed_counts"]["C6R"][m]}→{dist[c]["observed_counts"]["C7"][m]}' for m in ['nonzero_oscillation_proxy','answer_changes_present','any_retraction']]+[num(dist[c]['paired_deltas']['oscillation_proxy'][k]) for k in ['mean','median']]+[dist[c]['C7_higher_than_C6R']['oscillation_proxy']] for c in ['ALL','A','B','C','D']]), '',
    '全60条非零proxy两侧均48/60；proxy平均6.9→6.8，相邻claim变化存在22→17。退步组非零proxy7→9、撤回事件存在5→9，但claim变化存在5→2，proxy平均5.5→5.3、中位数1→2.5，6/10配对proxy上升。信号混合，不能据此断言“振荡导致10条退步”。正确组也大量存在非零proxy，说明“非零”不是有效失败分类器。', '',
    '重复指标复用4/8/16-token ngram：duplicate fraction=(总ngram数−唯一ngram数)/总ngram数；unique fraction=唯一数/总数，两者相加为1。高于配对C6R只表示相对变化，不设“重复失败”的阈值。', '',
    table(['类别','n','8gram median C6R %','C7 %','mean paired Δ pp','median paired Δ pp','C7重复度更高'], [[c,dist[c]['samples'],f'{100*dist[c]["conditions"]["C6R"]["repeated_8gram_fraction"]["median"]:.2f}',f'{100*dist[c]["conditions"]["C7"]["repeated_8gram_fraction"]["median"]:.2f}',f'{100*dist[c]["paired_deltas"]["repeated_8gram_fraction"]["mean"]:+.2f}',f'{100*dist[c]["paired_deltas"]["repeated_8gram_fraction"]["median"]:+.2f}',dist[c]['C7_higher_than_C6R']['repeated_8gram_fraction']] for c in ['ALL','A','B','C','D']]), '',
    '退步组4/8/16gram重复度分别7/10、6/10、6/10上升；8gram配对中位增幅+7.14个百分点，both-correct约−0.50个百分点，但both-wrong也上升约+6.48个百分点。这是较弱的错误轨迹关联，不是regression独有或因果机制。完整4/8/16及unique distributions见 `analysis/repetition_diagnostics.json`。', '',
    '## 7. First commitment 与不可辨识项', '',
    '旧CSV的 `first_committed_claim` 是整数答案值，不是位置；`first_committed_claim_char` 才是字符位置。`post_commit_fraction` 是字符比例，不能称token比例。字符位置现有可用C6R55/60、C748/60；退步组C6R10/10、C74/10。只报告既有character proxy分布，缺失不填0。', '',
    'first_commit_token、post_commit_token_count、final_retraction_present、semantic_answer_flip和long_repeated_spans均标为 `NOT_AVAILABLE_FROM_EXISTING_ARTIFACTS`。冻结摘要没有可靠token-char对齐、最终撤回专属标签或验证过的语义/长重复跨度分类；没有为本任务创造新提取规则。', '',
    '注意：旧trajectory CSV的raw-text claim枚举与正式评分的预处理路径不同；退步组6条proxy committed_claim_count=0，但正式V4只有3条ABSTAIN。必须以冻结V4评分定义弃答，不可把这6条重新标成弃答。', '',
    '## 8. 30-question × 2-seed结构', '',
    '10条regressions分布在8道题：02、05、09、11、12、22、26、27；11和27双seed退步，共4/10条；其余6题各1条。3条rescues分布在04、13、22，无双seed rescue。', '',
    table(['汇总','数量'],[['至少一seed regression题',8],['双seed regression题',2],['至少一seed rescue题',3],['双seed rescue题',0],['mixed seed题',8],['一seed rescue、另一seed regression题',1]]), '',
    'mixed定义：两seed类别不同且至少一个属于A/B。aime26_22表现为seed1 rescue、seed2 regression；它直接展示condition优势随seed变化。结构是“2道稳定失败+6道seed-sensitive失败”，不是几道稳定问题解释全部退步，也不是完全无question结构。逐30题表见 `analysis/question_cluster_stability.csv`。', '',
    '## 9. Historical Hadamard背景配对', '',
    'H的problem_id/seed/gold严格对齐60/60，来源hash验证通过。C6R rescue H=5、regression H=6、both-correct=35、both-wrong=14，净−1，11条discordance抵消。不是几乎完全相同的正确集合。', '',
    '标签 `HISTORICAL_CONTEXTUAL_ONLY`：历史H41/60（68.33%）、FP52/60（86.67%）、C636/60、C533/60、Native INT818/60只作背景，不是本次同runtime primary比较，不重跑baseline，不新增统计检验。H全部配对IDs见 `analysis/c6r_vs_h_contextual_taxonomy.json`。', '',
    '## 10. 必答 Q1–Q7', '',
    '**Q1：10条退步由什么直接解释？** 3条Frozen V4弃答，7条提取答案错误；length-hit=0，未出现EOS=0。9条非零oscillation proxy、2条claim-value变化只能报告代理存在；真正推理振荡、semantic answer flip、最终撤回机制不能直接确认。不能把代理计数当成已证实失败原因。', '',
    '**Q2：净增5条弃答是否主要落在regressions？** 不能把净增视为5个新增样本。真实流量7新增、2解除；7新增中3 regression、4 both-wrong。两条解除弃答仍both-wrong。弃答不是多数10条退步的解释。', '',
    '**Q3：稳定question failure还是seed-sensitive？** 两种都有，偏向seed-sensitive的后续诊断：8道退步题中2道双seed稳定，6道仅一个seed；aime26_22两seed方向相反。不足以称STRONG统一question-level机制。', '',
    '**Q4：3条rescues共同模式？** 04:seed2、13:seed1、22:seed1均变短（Δ−3321、−42205、−865），proxy下降（74→45、9→2、15→9），撤回事件减少；均无弃答/上限问题，两侧EOS。重复度并非一致改善：8/16gram两条下降，一条上升。只是3条的描述性共同模式，不能外推特定题型或训练因果。', '',
    '**Q5：C6R为什么从historical C6的36/60到40/60？** `NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS`。这些endpoint数据没有隔离训练、硬件、materialization或runtime差异的因果对照。+4是背景点估计，不归因于某个改动。', '',
    '**Q6：C6R和H只差1，正确集近乎相同吗？** 不是。5条rescue和6条regression抵消，共11条discordance。只作为严格ID对齐后的历史背景taxonomy。', '',
    '**Q7：下一类最支持什么？** 选择B `TRAJECTORY_TERMINATION_DIAGNOSIS_NEXT`，强度仅为小型后续诊断假设：弃答有增量、seed敏感及重复度关联可定位到固定discordants，值得先做shared-prefix replay。全局proxy未恶化、7条非弃答regression机制仍未知；不宣称已经解释C7退化，不因此选择新loss或layer-aware C8。若未来受控replay不支持该假设，应停止拓展而不是追加训练。', '',
    '## 11. Qualitative verdict解释', '',
    '- Termination/abstain：PRESENT（3/10 regression弃答；全部EOS、零length-hit，真正非EOS/长度终止失败为NOT_SUPPORTED）。',
    '- Trajectory instability：PRESENT（6题single-seed regression及重复度配对差；振荡/claim变化证据混合，不是STRONG）。',
    '- Question stability：PRESENT（两题双seed稳定，但不占8道退步题多数，不是STRONG）。',
    '- 没有endpoint级state/logit/layerwise错误轨迹，因此不从Ling结果推导Qwen layer-aware路线。C6R保留默认anchor，C7 Final-Hidden objective暂不拓展。', '',
    '## 12. 仅提案：25条固定discordant replay panel', '',
    '`QWEN_C6R_C7_DISCORDANT_TRAJECTORY_REPLAY_V1`：全部13 discordants+6 both-correct+6 both-wrong，controls按(problem_id升序, seed升序)各组取前6，不使用replay outcome。配置与逐ID在 `proposals/discordant_replay_panel.json`。', '',
    '- Regression：C6R正确trajectory作为共同teacher-forced tokens。Rescue：C7正确trajectory。两类controls固定C6Rtrajectory（both-wrong不声称reference正确）。',
    '- 同一prefix比较FP/H/C6R/C7，未来可测state/functional/final-hidden error、KL(FP||condition)、target ΔNLL、FP-reference top1、logit margin、EOS probability/margin和layerwise/tokenwise走势。',
    '- first-divergence proposal：使用已保存tokens确定首次分叉，仅在共同prefix t<divergence比较分叉前分布，不用condition-specific prefixes混淆机制。',
    '- 所有这些measurement均 `NOT_RUN`、authorization=NONE。须用户另行批准才能实施；本任务不运行replay，不设计并启动新GPU workload。', '',
    '## 13. Blindness披露及科学停止点', '',
    '`FULLY_BLINDED_EVALUATION=NO`：用户授权在endpoint完成前查看correctness/历史对比。固定60/condition仍全部完成、raw冻结、V4评分和hash闭环有效；不能称全程blind confirmatory，也不扩大成结果无效。旧endpoint报告中关于首次读历史结果时点的句子由已保存protocol amendments和本披露更正；源报告只读保留。', '',
    '`NEW_C8_TRAINING_AUTHORIZED=NO`，`DO_NOT_RUN_REPLAY=YES`，`FINAL_ACTION=STOP`。本报告不给出因果结论或新的confirmatory test，只给下一步推荐。不修改原Frozen V4、baseline、checkpoint或rotation。', '',
    '## 14. 产物与复现', '',
    '所有规定JSON/CSV已生成。`analysis/final_validation.json`记录独立读取冻结scores的60配对/30×2/13discordant/25panel复核与输入hash不变；`SHA256SUMS`覆盖此新实验所有文件（自身除外）。', '',
    '代码、完整再现命令及安全输出规则见 `REPRODUCE.md`；统计数据字典见 `DATA_DICTIONARY.md`。原始生成文本没有复制到新实验，CSV含问题文本及诊断值，不新增科研大tensor。', '']
    emit(d, 'FINAL_REPORT.md', '\n'.join(report))
    emit(d, 'DATA_DICTIONARY.md', '''# 数据字典与口径

- 一行一 problem_id+seed；sample_id 格式 aime26_XX:seedY。
- A=rescue，B=regression，C=both-correct，D=both-wrong。
- C6R/C7_correct、abstain、EOS、length_hit 是冻结V4/raw metadata的布尔值，不重新打分。
- generation_tokens/generated_tokens 是既有生成token数；delta_generation_length=C7−C6R。
- first_committed_claim 是既有整数答案值；first_committed_claim_char是字符位置；post_commit_fraction是字符比例。
- NA字符串 NOT_AVAILABLE_FROM_EXISTING_ARTIFACTS 表示不可可靠计算，不等于false/0；CSV空白表示既有proxy字段缺失，汇总n记录有效观测数。
- answer_changes/ABA/retractions/oscillation均复用旧定义，只是文本proxy；不与正式V4ABSTAIN重新互相推导。
- answer_flip_present_proxy 是answer_changes>0；semantic_answer_flip_present不可得。
- repetition分类只比较配对数值；4/8/16gram无需新增阈值，long spans不可得。
- distribution p25/p75使用(n−1)q线性插值；missing不填零。
- 原格式CSV重复度以0~1fraction存储，报告Δpp=100×fraction difference。
- 所有posthoc比较为descriptive；30道题×2seed，未新增任何显著性检验。
''')
    scripts = d/'scripts'; scripts.mkdir(exist_ok=True)
    check(not (scripts/'postendpoint_diagnosis.py').exists(), 'builder copy exists')
    shutil.copyfile(a.builder, scripts/'postendpoint_diagnosis.py')
    shutil.copyfile(__file__, scripts/'finalize_postendpoint_diagnosis.py')
    emit(d, 'REPRODUCE.md', f'''# CPU-only复现

Python: /data/ydai/miniconda3/envs/bitdecode/bin/python
Source endpoint: {src}
Endpoint git commit: {inventory['endpoint_commit']}

下面是本轮实际执行的命令。新输出仅限同级独立实验目录，builder拒绝覆盖已有目录。

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 19 /data/ydai/miniconda3/envs/bitdecode/bin/python /tmp/qwen_postendpoint_diagnosis_20261005.py --endpoint {src} --output {d}
PYTHONDONTWRITEBYTECODE=1 nice -n 19 /data/ydai/miniconda3/envs/bitdecode/bin/python /tmp/qwen_finalize_postendpoint_diagnosis_20261005.py --output {d} --builder /tmp/qwen_postendpoint_diagnosis_20261005.py
```

归档scripts是上述两份脚本的逐字副本。仅标准库，不import torch/transformers，不调用模型、scorer、nvidia-smi或GPU接口。因为指定实验输出已经存在，不能直接再次写入；若未来需要重复计算，应在另一个安全checkout复用相同source/hashes并独立存放输出，不能删除/覆盖本次产物。

SHA256SUMS以本实验根目录为基准相对路径，Linux验证：sha256sum -c SHA256SUMS。
脚本生成manifest前会再次验证所有consumed inputs与原endpoint final manifest。
''')
    validation = dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),independent_score_reproduction=True,paired_rows=60,questions=30,seeds_per_question=2,taxonomy_counts={c:len(grouped[c]) for c in 'ABCD'},C7_regression_abstain=3,C7_unique_abstain=7,C6R_unique_abstain=2,stable_regression_questions=stable,replay_panel_count=25,replay_panel_status='NOT_RUN',endpoint_manifest_files_checked=len(protected),all_consumed_input_hashes_unchanged=True,original_artifact_writes=0,new_model_forward=0,new_GPU_measurement=0,new_generation=0,new_training=0,scorer_calls=0,new_confirmatory_tests=0,report_Q1_Q7_present=True,final_action='STOP')
    emit(d,'analysis/final_validation.json',json.dumps(validation,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # Final read-only source recheck immediately before artifact hash closure.
    for p, digest in reval['consumed_source_hashes'].items(): check(sha(p) == digest, 'source final recheck '+p)
    for line in (src/'manifests/final_artifact_sha256.txt').read_text().splitlines():
        digest,name=line.split(None,1); p=Path(name.strip()); p=p if p.is_absolute() else src/p
        check(sha(p) == digest,'endpoint final recheck '+str(p))
    files = sorted(p for p in d.rglob('*') if p.is_file())
    emit(d,'SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(d).as_posix()}\n' for p in files))
    check(sum(p.stat().st_size for p in d.rglob('*') if p.is_file()) < 1<<30, 'unexpected output size')
    print(json.dumps(dict(status='PASS',files_hashed=len(files),artifact_bytes=sum(p.stat().st_size for p in d.rglob('*') if p.is_file()),report=str(d/'FINAL_REPORT.md'),manifest_sha256=sha(d/'SHA256SUMS'),final_action='STOP')))

if __name__ == '__main__':
    main()
