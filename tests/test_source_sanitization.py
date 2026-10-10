from bs4 import BeautifulSoup
from scripts.source_data import sanitize_html


def test_redacts_embedded_key_without_changing_policy_text():
    key = b'AIza' + b'X' * 35
    source = b'<script>key="' + key + b'"</script><main><p>Keep data for 30 days.</p></main>'
    sanitized = sanitize_html(source)
    assert key not in sanitized
    assert b'REDACTED_GOOGLE_API_KEY' in sanitized
    assert BeautifulSoup(source, 'html.parser').main.get_text() == BeautifulSoup(sanitized, 'html.parser').main.get_text()
    assert sanitize_html(sanitized) == sanitized
