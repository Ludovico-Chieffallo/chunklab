def test_markdown_loader_text_and_elements(handbook):
    assert "Overtime is paid at 1.5x" in handbook.text
    headings = [e for e in handbook.elements if e.type == "heading"]
    tables = [e for e in handbook.elements if e.type == "table"]
    assert any(h.text.startswith("Acme Corporation") and h.level == 1 for h in headings)
    assert any(h.level == 3 for h in headings)
    assert len(tables) == 1
    assert "Pension match" in tables[0].text


def test_element_spans_match_source(handbook):
    for el in handbook.elements:
        s, e = el.char_span
        assert el.text in handbook.text[s:e] or handbook.text[s:e].strip() == el.text
