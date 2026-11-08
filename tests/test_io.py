import io

import pandas as pd

from leakdet.io import parse_leak_config, parse_sensor_lists, read_scada_csv


def test_read_scada_csv_decimal_comma():
    text = "Timestamp;n1;n4\n2019-01-01 00:00:00;28,92;33,87\n2019-01-01 00:05:00;28,94;-1,5\n"
    df = read_scada_csv(io.StringIO(text))
    assert list(df.columns) == ["n1", "n4"]
    assert df.loc["2019-01-01 00:05:00", "n4"] == -1.5
    assert df.index[1] - df.index[0] == pd.Timedelta(minutes=5)


def test_parse_leak_config(leaks):
    assert len(leaks) == 33
    by = {lk.link: lk for lk in leaks}
    assert by["p257"].start == pd.Timestamp("2018-01-08 13:30")
    assert by["p257"].kind == "incipient"
    assert by["p673"].kind == "abrupt" and by["p673"].peak == by["p673"].start
    assert abs(by["p427"].diameter_m - 0.0090731) < 1e-12
    assert sum(lk.start.year == 2019 for lk in leaks) == 19


def test_parse_sensor_lists():
    from conftest import FIX

    lists = parse_sensor_lists((FIX / "dataset_configuration.yaml").read_text(encoding="utf-8"))
    assert len(lists["pressure_sensors"]) == 33
    assert lists["flow_sensors"] == ["PUMP_1", "p227", "p235"]
    assert lists["level_sensors"] == ["T1"]
    assert len(lists["amrs"]) == 82


def test_leak_line_with_spaces_is_parsed():
    text = "leakages:\n- p31,  2018-06-28 10:35, 2018-08-12 17:30, 0.016389, incipient, 2018-08-03 02:45\n"
    (lk,) = parse_leak_config(text)
    assert lk.link == "p31" and lk.end == pd.Timestamp("2018-08-12 17:30")
