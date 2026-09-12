from __future__ import annotations

import wandb

from rxnhaystack.campaign_metrics import install_campaign_metrics


def main() -> None:
    install_campaign_metrics(wandb)
    run = wandb.init(project="rxnhaystack-infrastructure", mode="disabled", config={})
    wandb.log(
        {
            "sample/0/iteration_total_tokens": 12,
            "sample/0/final_total_input_tokens": 8,
            "sample/0/final_total_output_tokens": 4,
            "sample/0/final_total_tokens": 12,
            "sample/0/final_total_cost_usd": 0.0,
        }
    )
    run.summary["smoke_passed"] = True
    wandb.finish()


if __name__ == "__main__":
    main()
