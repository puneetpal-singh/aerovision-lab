def test_whole_mode_dispatch_and_cpu_default(monkeypatch):
    from aerovision import cli, experiment

    captured = {}
    monkeypatch.setattr(experiment, "compare", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        "sys.argv",
        ["aerovision", "compare", "--data", "sample.yaml", "--weights", "best.pt", "--mode", "whole"],
    )
    cli.main()
    assert captured["modes"] == ("whole",)
    assert captured["device"] == "cpu"
