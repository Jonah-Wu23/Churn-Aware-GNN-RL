import torch

from src.train.evaluation_checkpointer import EvalConfig, EvaluationCheckpointer


def test_eval_should_trigger_without_exact_modulo() -> None:
    cfg = EvalConfig(eval_interval_steps=5000)
    checkpointer = EvaluationCheckpointer(
        config=cfg,
        env_factory=lambda: None,
        select_action_fn=lambda _model, _features: None,
        compute_rho_fn=lambda _info: 0.0,
        device=torch.device("cpu"),
        gamma=1.0,
    )

    assert checkpointer.should_evaluate(0) is False
    assert checkpointer.should_evaluate(4999) is False
    assert checkpointer.should_evaluate(5000) is True
    assert checkpointer.should_evaluate(6000) is True

    checkpointer._last_eval_global_step = 6000
    assert checkpointer.should_evaluate(7000) is False
    assert checkpointer.should_evaluate(10999) is False
    assert checkpointer.should_evaluate(11000) is True

