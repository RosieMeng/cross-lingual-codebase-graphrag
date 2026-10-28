"""Evidence-first answers for unambiguous graph relationship questions."""
from __future__ import annotations

import re
from .multilingual_query import detect_language


def answer_language(question):
    """Honor narrow explicit language requests, otherwise use existing detection."""
    text = question.casefold()
    if re.search(r"\b(?:in french|en fran[çc]ais)\b|用法语|法语回答", text): return "fr"
    if re.search(r"\b(?:in chinese|en chinois)\b|用中文|中文回答", text): return "zh"
    if re.search(r"\b(?:in english|en anglais)\b|用英语|英文回答", text): return "en"
    detected = detect_language(question)
    if detected.language in {"en", "fr", "zh"}: return detected.language
    if detected.language == "mixed":
        if "zh" in detected.languages: return "zh"
        if "en" in detected.languages: return "en"
        if "fr" in detected.languages: return "fr"
    if re.search(r"\b(?:qui appelle|quelles? fonctions? appellent|qu['’]appelle)\b", text): return "fr"
    return "en"


def graph_answer(question, info, evidence):
    """Return a source-bound answer, or None for non-relationship requests.

    Never use unresolved call names as proof of a runtime relationship.
    """
    direction = info.get("direction")
    if not info or direction not in {"callers", "callees"}:
        return None
    language = answer_language(question)
    strings = {
        "en": {"missing": "I cannot identify one source-backed target for this call question. Please provide an exact qualified symbol and, if needed, its file.",
               "root": "Definition", "static": "Static reference (runtime dispatch not verified)",
               "unresolved": "Observed call site; target unresolved", "none": "No resolved references were found in the available graph; it may be incomplete.",
               "omitted": "Some call sites are not shown because their source was not retrieved or the graph results were limited."},
        "fr": {"missing": "Je ne peux pas identifier une cible unique étayée par le code. Indiquez un symbole qualifié exact et, si nécessaire, son fichier.",
               "root": "Définition", "static": "Référence statique (cible à l’exécution non vérifiée)",
               "unresolved": "Appel observé ; cible non résolue", "none": "Aucune référence résolue dans le graphe disponible ; celui-ci peut être incomplet.",
               "omitted": "Certains appels ne sont pas affichés car leur source n’a pas été récupérée ou les résultats du graphe sont limités."},
        "zh": {"missing": "无法根据现有代码证据确定唯一的调用目标。请提供准确的限定符号名，必要时提供文件路径。",
               "root": "定义", "static": "静态引用（未验证运行时目标）",
               "unresolved": "已观察到调用位置；目标未解析", "none": "现有图中没有解析出的引用；图可能不完整。",
               "omitted": "部分调用位置因未检索到源码或图结果数量受限而未显示。"},
    }[language]
    root = info.get("root")
    if info.get("lookup_status") != "matched" or not root:
        return strings["missing"]
    by_id = {item.get("id"): item for item in evidence}
    by_citation = {item.get("citation_id"): item for item in evidence}
    root_evidence = by_id.get(root.get("chunk_id"))
    if (not root_evidence or not root_evidence.get("citation_id") or root_evidence.get("file") != root.get("file")
            or not isinstance(root_evidence.get("line_start"), int) or not isinstance(root_evidence.get("line_end"), int)
            or not root_evidence["line_start"] <= root["line_start"] <= root["line_end"] <= root_evidence["line_end"]):
        return strings["missing"]
    lines = [f"{strings['root']}: `{root['qualified']}` — {root['file']}:{root['line_start']}-{root['line_end']} [{root_evidence['citation_id']}]" ]

    def source(row, file_key, line_key):
        citation = row.get("caller_citation_id")
        item = by_citation.get(citation)
        if not item or item.get("id") != row.get("caller_chunk_id") or item.get("file") != row.get(file_key): return None
        start, end, line = item.get("line_start"), item.get("line_end"), row.get(line_key)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end, line)) or not start <= line <= end: return None
        return citation

    omitted = False
    for edge in info.get("relationships", []):
        citation = source(edge, "caller_file", "call_line")
        target_citation = edge.get("callee_citation_id")
        target = by_citation.get(target_citation)
        if (edge.get("status") != "static_reference" or not citation or not target
                or target.get("id") != edge.get("callee_chunk_id") or target.get("file") != edge.get("callee_file")
                or not isinstance(target.get("line_start"), int) or not isinstance(target.get("line_end"), int)
                or not isinstance(edge.get("callee_line_start"), int) or not isinstance(edge.get("callee_line_end"), int)
                or not target["line_start"] <= edge["callee_line_start"] <= edge["callee_line_end"] <= target["line_end"]):
            omitted = True
            continue
        lines.append(f"- {strings['static']}: `{edge['caller']}` → `{edge['callee']}`; `{edge['expression']}` — {edge['caller_file']}:{edge['call_line']} [{citation}]; {edge['callee_file']}:{edge['callee_line_start']}-{edge['callee_line_end']} [{target_citation}]")
    for call in info.get("unresolved", []):
        citation = source(call, "file", "line")
        if call.get("status") != "unresolved_target" or not citation:
            omitted = True
            continue
        lines.append(f"- {strings['unresolved']}: `{call['expression']}` — {call['file']}:{call['line']} [{citation}]")
    if not info.get("relationships"):
        lines.append(strings["none"])
    if (omitted or info.get("relationships_total", 0) > len(info.get("relationships", []))
            or info.get("unresolved_total", 0) > len(info.get("unresolved", []))):
        lines.append(strings["omitted"])
    return "\n".join(lines)
