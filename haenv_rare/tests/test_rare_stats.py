"""Text metrics of `haenv-rare-stats`: counted the same way for real and synthetic reports."""
from haenv_rare.stats import _en_case_section, _zh_case_section, text_metrics


def test_zh_metrics():
    md = "## 主诉\n患者男性，因“癫痫发作”就诊。\n## 体格检查\n未见皮疹。心肺无异常。其母亲有类似病史。\n"
    m = text_metrics(md, "zh")
    assert (m["sentences"], m["negated"], m["subsections"]) == (4, 2, 2) and not m["family"]
    assert text_metrics("家族史：否认。", "zh")["family"]


def test_en_metrics():
    md = "## Exam\nNo rash was seen. The heart was normal. His mother has a history of seizures.\n"
    m = text_metrics(md, "en")
    assert (m["sentences"], m["negated"], m["subsections"], m["family"]) == (3, 1, 1, True)


def test_case_sections_stop_at_the_next_top_level_heading():
    zh = "# 题目\n## 1 临床资料\n### 1.1 现病史\n患者。\n## 2 讨论\n讨论。"
    assert _zh_case_section(zh).strip() == "### 1.1 现病史\n患者。"
    en = "## Introduction\nx\n## Case presentation\nA boy.\n### Labs\nok\n## Discussion\ny"
    assert _en_case_section(en).strip() == "A boy.\n### Labs\nok"
