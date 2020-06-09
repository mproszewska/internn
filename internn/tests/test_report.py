import pytest

import internn as inn


@pytest.mark.parametrize("display", [True, False])
def test_report_parameters(
    capsys, display,
):
    msg = "test=0\n0=test\n"
    reporter = inn.Reporter(display)
    reporter.report_parameters({"test": 0, 0: "test"})

    captured = capsys.readouterr()

    if display:
        assert captured.out == msg
    else:
        assert captured.out == ""


@pytest.mark.parametrize("display", [True, False])
def test_report_message(capsys, display):
    msg = "test message"
    reporter = inn.Reporter(display=display)
    reporter.report_message(msg)

    captured = capsys.readouterr()

    if display:
        assert captured.out == msg
    else:
        assert captured.out == ""
