from streamlit.testing.v1 import AppTest
from policydiff.settings import ROOT


def test_local_sample_comparison_and_settings_invalidate_report():
    app = AppTest.from_file(ROOT / 'app.py').run(timeout=30)
    assert not app.exception
    app.sidebar.selectbox[0].select('Textual · difflib')
    app.radio[0].set_value('Try a sample').run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 6
    counts = app.session_state['report']['counts']
    assert counts['modified'] + counts['reworded'] >= 2
    app.sidebar.selectbox[0].select('Lexical · TF-IDF').run()
    assert len(app.metric) == 0
