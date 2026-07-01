"""Auto-resume logic for interrupted pipeline runs."""

import logging
from typing import Optional, List, Dict, Any

from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.config_hash import ConfigFingerprint

logger = logging.getLogger(__name__)


class ResumeManager:
    """Determines which stages need re-running and validates artifacts."""

    def __init__(self, dataset: str, output_base: str = "outputs"):
        self.dataset = dataset
        self.output_base = output_base
        self.ck = CheckpointManager(dataset, output_base)
        self.fp = ConfigFingerprint(dataset, output_base)

    def plan_resume(
        self,
        config_path: str,
        skip_stages: Optional[List[str]] = None,
        force_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        skip_stages = skip_stages or []
        force_stages = force_stages or []

        completed = self.ck.completed_stages()
        config_valid = self.fp.validate_fingerprint(config_path)
        if not config_valid:
            logger.warning("Config fingerprint changed - invalidating all stages")
            self.ck.clear_all()
            completed = []

        to_run: List[str] = []
        for stage in self.ck.STAGES:
            if stage in skip_stages and self.ck.stage_done(stage):
                logger.info(f"Stage '{stage}' already done, skipping")
                continue
            if stage in force_stages:
                logger.info(f"Force re-running stage '{stage}'")
                self.ck.clear_stage(stage)
                to_run.append(stage)
                continue
            if stage in completed:
                artifacts_ok = self.ck.validate_stage_artifacts(stage)
                missing = [path for path, exists in artifacts_ok.items() if not exists]
                if missing:
                    logger.warning(
                        "Stage '%s' marked done but missing %d artifact(s): %s",
                        stage, len(missing), missing,
                    )
                    self.ck.clear_stage(stage)
                    to_run.append(stage)
                else:
                    logger.info("Stage '%s' OK — all artifacts present", stage)
            else:
                to_run.append(stage)

        self.fp.save_fingerprint(config_path, stage="resume_plan")

        return {
            "completed": completed,
            "to_run": to_run,
            "config_changed": not config_valid,
        }

    def stage_start(self, stage: str):
        logger.info(f"Starting stage: {stage}")

    def stage_complete(self, stage: str, metadata: Optional[Dict[str, Any]] = None):
        self.ck.mark_done(stage, metadata)
        logger.info(f"Completed stage: {stage}")

    def reset_from(self, stage: str):
        idx = self.ck.STAGES.index(stage) if stage in self.ck.STAGES else -1
        if idx >= 0:
            for s in self.ck.STAGES[idx:]:
                self.ck.clear_stage(s)

    def resume_summary(self) -> str:
        completed = self.ck.completed_stages()
        lines = [f"Dataset: {self.dataset}"]
        lines.append(f"Completed: {len(completed)}/{len(self.ck.STAGES)}")
        for s in self.ck.STAGES:
            marker = "+" if s in completed else "o"
            lines.append(f"  {marker} {s}")
        return "\n".join(lines)
