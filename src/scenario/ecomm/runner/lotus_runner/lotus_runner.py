"""
Lotus system runner implementation.
Placeholder required by the current structure of the benchmarking framework.
"""

# ============================================================
# 教学注释 (L1 wrapper pass):
# ------------------------------------------------------------
# E-commerce scenario 的 Code* mode wrapper. 比 cars / medical 复杂得多:
#
#   1) Cascade wiring (line 37-45) : 如果 policy="approximate", 实例化
#      SentenceTransformersRM (text + image 双 RM) + FaissVS + CascadeArgs.
#      paper §3 cascade 的实际启用入口.
#
#   2) Modality-aware join wiring (_configure_lotus_for_join_type, line 47-62) :
#      根据 join_type ("text" / "image" / "mixed") 切换 RM 用哪种 embedding.
#      - text only → e5-base-v2 (纯文本 768-d)
#      - image / mixed → clip-ViT-B-32 (能 encode 图也能 encode 文)
#      query 文件内通过 _configure_lotus(join_type) 调用.
#
#   3) Helper injection (line 86-89) : exec() 之前在 query_module 命名空间
#      注入 _configure_lotus / _cascade_args / _policy. 让 Q<i>.py 内代码
#      可以直接调 _configure_lotus("image") 等. 这是 SemBench 给 Code* mode
#      query 暴露 cascade 设置的机制 — 否则 query 文件没法访问 runner instance.
#
# 默认 model "vertex_ai/gemini-2.5-flash" (line 29) — 注意 LiteLLM provider
# prefix "vertex_ai/" 不能省 ([LOTUS/LOG.md pitfall #6]).
#
# 路径不一致警告 ([LOTUS/LOG.md 关键发现 #8]):
#   ecomm 用 files/ecomm/queries/dialects/lotus/q{i}.py (queries + dialects
#   多两层); 其它 scenario 用 files/{sc}/query/lotus/Q{i}.py. scenario_handler
#   内部处理这个差异.
# ============================================================
from pathlib import Path
import sys
import time
import types
import lotus
import traceback

from runner.generic_runner import GenericQueryMetric, GenericRunner

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_lotus_runner.generic_lotus_runner import GenericLotusRunner

# Import additional modules for approximate policy
# SentenceTransformersRM : 给 cascade helper 当 proxy (cosine sim 当 proxy_score)
# CascadeArgs : 阈值参数 (paper §3, 详 [AllSQPE/LOTUS/types.py:163])
# FaissVS : 默认 local vector store
from lotus.models import SentenceTransformersRM
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS


class LotusRunner(GenericLotusRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "vertex_ai/gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker
        )

        # Initialize components for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            # Initialize both embedding models for mixed modality support
            self.rm_text = SentenceTransformersRM(model="intfloat/e5-base-v2")
            self.rm_image = SentenceTransformersRM("clip-ViT-B-32")
            self.vs = FaissVS()
            self.cascade_args = CascadeArgs(
                recall_target=0.8, precision_target=0.8
            )

    # ========================================================
    # _configure_lotus_for_join_type: per-query modality switch
    # --------------------------------------------------------
    # ecomm 的 query 涉及不同模态 join:
    #   - Q1-Q5 主要 text-text (product description vs review)
    #   - Q6-Q10 含 image (product image vs query)
    #   - Q11-Q14 mixed (text + image)
    # 不同模态需要不同 RM:
    #   - text RM (e5-base-v2) : 处理纯文本 sim, 不识别图
    #   - image RM (clip-ViT-B-32) : 能 cross-modal (text 与 image cosine sim)
    # 这里给 query 内代码暴露切换接口, query 在 cascade 之前调
    #   _configure_lotus("image") 或类似, settings.rm 就被换.
    #
    # 注: 此函数 lazy 切换 — 不 sem_search 时不切; sem_search 之前必须调.
    # 否则 settings.rm 留着上次的, query 跑出错或精度低.
    # ========================================================
    def _configure_lotus_for_join_type(self, join_type: str):
        """Configure LOTUS settings based on join type (text-only, image-only, or mixed)."""
        if hasattr(self, "policy") and self.policy == "approximate":
            if join_type == "text":
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_text, vs=self.vs
                )
            elif join_type == "image":
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_image, vs=self.vs
                )
            else:  # mixed or default
                # For mixed modality, use image embeddings as they handle both
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_image, vs=self.vs
                )

    def _discover_queries(self):
        # Match default implementation from GenericRunner
        return GenericRunner._discover_queries(self)

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        # Reset token stats before each query
        try:
            lotus.settings.lm.reset_stats()
        except Exception as e:
            print(f"  Warning: Could not reset stats: {e}")

        try:
            # The queries in LOTUS are Python files with a run() function.
            # Load its contents, create a module, invoke the run() function.
            query_text = self.scenario_handler.get_query_text(
                query_id, self.get_system_name()
            )
            query_module = types.ModuleType(f"q{query_id}_module")

            # Inject helper functions and objects for approximate policy
            if hasattr(self, "policy") and self.policy == "approximate":
                query_module.__dict__["_configure_lotus"] = self._configure_lotus_for_join_type
                query_module.__dict__["_cascade_args"] = self.cascade_args
                query_module.__dict__["_policy"] = self.policy

            exec(query_text, query_module.__dict__)

            start_time = time.time()
            results = query_module.run(self.scenario_handler.get_data_dir())
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results
            metric.status = "success"

            # Get token usage and cost
            self._update_token_usage(metric)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        finally:
            # Reset stats after storing
            try:
                lotus.settings.lm.reset_stats()
            except:
                pass

        return metric
