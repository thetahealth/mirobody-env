"""joint_scenarios.py -- data spec for joint_dx differential-diagnosis (ddx) cases.

Each entry is one patient's different symptoms at different times plus a
single answer (diagnosis / tests / specialty / urgency / join_gold). The answer
goes into `adjudication` (verifier-only); the symptoms become solver-visible
evidence. 7 entries are single diseases with multi-system presentations
(unified); 10 are 8 unified great imitators, 1 comorbidity and 1 independent.
SYNTHETIC, evaluation use only, not medical advice.
"""
from __future__ import annotations

# join_gold: unified (same underlying process) / comorbidity (>=2 independent processes) /
#            independent (mutually independent, an over-merging trap)
# weight: (kind, start, end) -- a nominal metabolic background signal (obesity domain, [55,110]);
#         the real content is in symptoms
# urgency: 🔴 immediate ER / 🟠 within days / 🟡 routine evaluation soon / 🟢 can be observed
# symptoms: [(day<=84, symptom, context)] -- different symptoms across time
DDX_SPECS: dict[str, dict] = {
    # ============================== 7 single-disease, multi-system (unified) ==============
    "JD-PCOS": {
        "diagnosis": "多囊卵巢综合征(PCOS)", "aliases": ["多囊", "pcos", "polycystic"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("up", 61.0, 62.5),
        "tests": ["性激素六项/睾酮·SHBG(FAI)", "经阴道超声(PCOM)或AMH", "OGTT+胰岛素", "排除TSH/PRL/17-OHP"],
        "specialty": ["妇科内分泌/生殖内分泌", "内分泌科"],
        "symptoms": [(14, "下颌线反复炎性痤疮", "红肿硬结,护肤/外用无明显效"),
                     (35, "月经稀发、周期推迟40+天", "无怀孕可能"),
                     (56, "半年增重,严格低碳+运动两周体重几乎不降", "代谢阻力"),
                     (77, "颈后对称发黑增厚(黑棘皮)、去角质无效", "")],
    },
    "JD-CUSH": {
        "diagnosis": "库欣综合征", "aliases": ["库欣", "cushing", "皮质醇增多"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("up", 74.0, 76.0),
        "tests": ["24h尿游离皮质醇", "1mg地塞米松抑制试验", "午夜唾液皮质醇", "ACTH+垂体/肾上腺影像"],
        "specialty": ["内分泌科"],
        "symptoms": [(14, "向心性肥胖:躯干变胖而四肢变细、腰围增", ""),
                     (35, "35岁新发高血压≈148/95,严格低盐一月无改善", ""),
                     (56, "空腹血糖6.9 + 入睡困难、情绪低落", ""),
                     (77, "下腹/大腿内侧宽紫红纹 + 易瘀青 + 近端肌无力", "复查提示低钾倾向")],
    },
    "JD-HYPO": {
        "diagnosis": "原发性甲状腺功能减退(桥本)", "aliases": ["甲减", "甲状腺功能减退", "hypothyroid", "桥本"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("up", 71.0, 72.0),
        "tests": ["TSH", "FT4", "TPOAb"], "specialty": ["内分泌科", "内科"],
        "symptoms": [(14, "吃得更少却持续增重 + 显著怕冷", ""),
                     (35, "睡够8h仍极度乏力,静息心率56", "血常规无贫血"),
                     (56, "顽固便秘,增纤维+益生菌无效", ""),
                     (77, "月经量多经期延长 + 清淡饮食却血脂高、反应迟钝", "")],
    },
    "JD-OSA": {
        "diagnosis": "阻塞性睡眠呼吸暂停(OSA)", "aliases": ["osa", "睡眠呼吸暂停", "呼吸暂停"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("stable", 94.0, 94.0),
        "tests": ["多导睡眠监测(PSG)/AHI"], "specialty": ["睡眠医学科", "呼吸科"],
        "symptoms": [(14, "卧床8h+仍白天严重嗜睡、开会开车犯困", "手环夜间血氧掉到84-88%"),
                     (35, "晨起头痛,换高枕+睡前多喝水无效", ""),
                     (56, "血压三药联用仍≈148/95难控", ""),
                     (77, "夜尿3-4次(前列腺+血糖正常)+性欲下降", "家人目击打鼾中憋气—猛吸气")],
    },
    "JD-IR": {
        "diagnosis": "早期2型糖尿病/胰岛素抵抗", "aliases": ["糖尿病", "胰岛素抵抗", "diabetes", "insulin"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("up", 82.0, 83.0),
        "tests": ["空腹血糖/OGTT/HbA1c", "空腹胰岛素/HOMA-IR"], "specialty": ["内分泌科"],
        "symptoms": [(14, "颈后天鹅绒样发黑增厚,去角质一周无效", "BMI28久坐高糖"),
                     (35, "餐后明显困倦 + 口渴、多尿感", ""),
                     (56, "验光度数没变却视物时清时糊(甜食后加重)", ""),
                     (77, "反复口腔溃疡+皮肤疖子,补维生素无效", "")],
    },
    "JD-ALDO": {
        "diagnosis": "原发性醛固酮增多症", "aliases": ["原醛", "醛固酮", "conn", "aldosteron"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("stable", 80.0, 80.0),
        "tests": ["醛固酮/肾素比值(ARR)", "确诊试验(盐负荷等)", "肾上腺CT/AVS"],
        "specialty": ["高血压专科", "内分泌科"],
        "symptoms": [(14, "38岁难治高血压,严格低盐+三联降压仍≈146/96", "无家族史"),
                     (35, "乏力、爬楼腿软,活动后加重 + 手指麻木", ""),
                     (56, "夜间小腿抽筋,补钙镁只部分缓解", ""),
                     (77, "夜尿口渴(血糖正常)+ 自发性低钾3.1(补钾仅回到3.4)", "")],
    },
    "JD-HEMO": {
        "diagnosis": "遗传性血色病(铁过载)", "aliases": ["血色病", "铁过载", "hemochromat", "青铜"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("stable", 68.0, 68.0),
        "tests": ["血清铁蛋白", "转铁蛋白饱和度", "HFE基因检测", "肝铁定量MRI/活检"],
        "specialty": ["消化内科/肝病科", "血液科"],
        "symptoms": [(14, "不胖(BMI23)无家族史却新发糖尿病(空腹8.5,控糖仍高)", ""),
                     (35, "几乎不饮酒却ALT/AST持续高,B超非重度脂肪肝", ""),
                     (56, "第2、3掌指关节晨僵隐痛,热敷护腕无效", ""),
                     (77, "皮肤一年整体变灰暗(少晒、防晒无效)+ 性欲减退", "")],
    },
    # ============================== 10 imitators / comorbidity / independent ==============
    # --- 8 unified great-imitators ---
    "JD-GRAVES": {
        "diagnosis": "甲状腺功能亢进(Graves)", "aliases": ["甲亢", "graves", "甲状腺功能亢进"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("down", 66.0, 62.0),
        "tests": ["TSH", "FT4/FT3", "TRAb/TSI", "甲状腺摄碘率/超声"], "specialty": ["内分泌科"],
        "symptoms": [(14, "非意愿消瘦 + 心悸、手抖", ""),
                     (35, "怕热多汗、焦虑失眠", ""),
                     (56, "大便次数增多/稀便", ""),
                     (77, "眼睑退缩/凝视、月经稀少", "")],
    },
    "JD-PHEO": {
        "diagnosis": "嗜铬细胞瘤", "aliases": ["嗜铬细胞瘤", "pheochromocytoma", "pheo"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": True, "clinician_warranted": True, "weight": ("down", 74.0, 71.0),
        "tests": ["血/24h尿 甲氧基肾上腺素类(metanephrines)", "腹部CT/MRI", "MIBG(必要时)"],
        "specialty": ["内分泌科", "高血压专科"],
        "symptoms": [(14, "阵发性剧烈头痛 + 心悸 + 大汗三联", ""),
                     (35, "发作时血压骤升(220/120)、间歇正常", ""),
                     (56, "体重下降 + 惊恐/焦虑发作样", ""),
                     (77, "发作后面色苍白、乏力", "")],
    },
    "JD-ACRO": {
        "diagnosis": "肢端肥大症", "aliases": ["肢端肥大", "acromegaly", "生长激素"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("up", 84.0, 86.0),
        "tests": ["IGF-1", "OGTT-生长激素抑制试验", "垂体MRI"], "specialty": ["内分泌科"],
        "symptoms": [(14, "鞋码/戒指变大、手足增大", ""),
                     (35, "面容变粗、鼻唇增厚、多汗", ""),
                     (56, "新发糖尿病 + 高血压", ""),
                     (77, "多关节痛 + 打鼾 + 头痛", "")],
    },
    "JD-PHPT": {
        "diagnosis": "原发性甲状旁腺功能亢进(高钙血症)", "aliases": ["甲状旁腺", "甲旁亢", "高钙", "hyperparathyroid", "pth"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("stable", 70.0, 70.0),
        "tests": ["血钙(校正)", "PTH", "25-OH-VitD", "肾脏超声/尿钙"], "specialty": ["内分泌科"],
        "symptoms": [(14, "乏力、骨痛", ""),
                     (35, "肾结石发作/镜下血尿", ""),
                     (56, "多尿、口渴、便秘", ""),
                     (77, "情绪低落、记忆减退(stones/bones/groans/moans)", "")],
    },
    "JD-B12": {
        "diagnosis": "维生素B12缺乏(亚急性联合变性)", "aliases": ["b12", "维生素b12", "钴胺", "cobalamin"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟡",
        "red_flag": False, "clinician_warranted": True, "weight": ("stable", 72.0, 72.0),
        "tests": ["血清维生素B12", "同型半胱氨酸/甲基丙二酸(MMA)", "血常规MCV", "内因子抗体"],
        "specialty": ["血液科/神经内科", "内科"],
        "symptoms": [(14, "乏力、舌炎/口角炎", "大细胞性倾向"),
                     (35, "手脚对称麻木、针刺感", ""),
                     (56, "步态不稳、平衡变差", ""),
                     (77, "记忆减退、情绪改变", "")],
    },
    "JD-SLE": {
        "diagnosis": "系统性红斑狼疮(SLE)", "aliases": ["狼疮", "sle", "lupus", "系统性红斑"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("down", 60.0, 58.0),
        "tests": ["ANA", "抗dsDNA/抗Sm", "补体C3/C4", "尿蛋白/肌酐比"], "specialty": ["风湿免疫科"],
        "symptoms": [(14, "多关节痛、晨僵", ""),
                     (35, "面部蝶形红斑、光敏", ""),
                     (56, "反复低热、乏力、脱发、口腔溃疡", ""),
                     (77, "泡沫尿/下肢水肿(肾受累)", "")],
    },
    "JD-LADA": {
        "diagnosis": "成人隐匿性自身免疫糖尿病(LADA)", "aliases": ["lada", "自身免疫糖尿病", "1型", "gad"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("down", 70.0, 66.0),
        "tests": ["空腹血糖/HbA1c", "GAD抗体/IA-2抗体", "C肽", "尿酮体"], "specialty": ["内分泌科"],
        "symptoms": [(14, "多饮、多尿", ""),
                     (35, "非意愿体重明显下降", "BMI不高"),
                     (56, "乏力、视物模糊", ""),
                     (77, "口渴加重、偶有酮症倾向", "")],
    },
    "JD-ADDISON": {
        "diagnosis": "原发性肾上腺皮质功能减退(Addison)", "aliases": ["addison", "肾上腺皮质功能减退", "艾迪生"],
        "join_gold": "unified", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("down", 68.0, 64.0),
        "tests": ["晨血皮质醇", "ACTH", "ACTH兴奋试验", "电解质(低钠高钾)"], "specialty": ["内分泌科"],
        "symptoms": [(14, "进行性乏力、消瘦", ""),
                     (35, "体位性低血压、站起头晕", ""),
                     (56, "皮肤黏膜色素沉着加深", ""),
                     (77, "嗜盐、恶心、低钠倾向", "")],
    },
    # --- 1 comorbidity (>=2 independent processes overlapping) ---
    "JD-CKM": {
        "diagnosis": "心肾代谢综合征:2型糖尿病失控 + 慢性肾病进展", "aliases": ["糖尿病", "慢性肾病", "ckd", "心肾"],
        "join_gold": "comorbidity", "outcome_label": "event_occurred", "urgency": "🟠",
        "red_flag": False, "clinician_warranted": True, "weight": ("up", 88.0, 90.0),
        "tests": ["HbA1c/血糖", "eGFR/血肌酐", "尿白蛋白/肌酐(UACR)", "NT-proBNP/心超"],
        "specialty": ["内分泌科", "肾内科/心内科"],
        "symptoms": [(14, "多尿口渴、血糖持续高(糖尿病线程)", ""),
                     (28, "夜尿增多 + 泡沫尿(肾脏线程)", ""),
                     (56, "双下肢可凹性水肿 + 活动后气短(心肾)", ""),
                     (77, "乏力 + 血压升高", "两个过程叠加,非单一")],
    },
    # --- 1 independent (an over-merging trap) ---
    "JD-BENIGN2": {
        "diagnosis": "无统一病理:彼此独立的良性事件", "aliases": ["独立", "良性", "无统一", "unknown"],
        "join_gold": "independent", "outcome_label": "event_not_occurred", "urgency": "🟢",
        "red_flag": False, "clinician_warranted": False, "weight": ("stable", 72.0, 72.0),
        "tests": ["无需系统性排查;针对各自症状对症/观察即可"], "specialty": ["全科/自我观察"],
        "symptoms": [(14, "打球崴脚、右膝酸痛", "机械性,RICE 后好转"),
                     (35, "换季过敏性鼻炎、喷嚏清涕", "抗组胺缓解"),
                     (56, "熬夜后一次紧张性头痛", "休息后消退"),
                     (77, "吃辣后一次胃胀反酸", "饮食相关")],
    },
}

# Registration order for jd04..jd20
CASE_ORDER = ["JD-PCOS", "JD-CUSH", "JD-HYPO", "JD-OSA", "JD-IR", "JD-ALDO", "JD-HEMO",
              "JD-GRAVES", "JD-PHEO", "JD-ACRO", "JD-PHPT", "JD-B12", "JD-SLE",
              "JD-LADA", "JD-ADDISON", "JD-CKM", "JD-BENIGN2"]
