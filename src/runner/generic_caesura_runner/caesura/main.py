# =============================================================================
# SemBench L1 vendored 副本 — caesura/main.py
# =============================================================================
# 教学注释 pass (L1 MODIFY) by Claude.
# 这是 L0 [AllSQPE/CAESURA/caesura/main.py](../../../../../../AllSQPE/CAESURA/caesura/main.py) 的 vendored 副本,
# 含 3 处 patch (相对 L0 +21 行), 加 *per-query lifecycle* 支持 SemBench 跨 query 复用 agent。
#
# Patch 位置 (本文件):
#   1. line 41         : self.last_result = None       — __init__ 新字段
#   2. line 87-88      : self.llm.reset_token_usage()  — run() 开头 reset token 计数
#   3. line 104        : self.last_result = final_result — run() finally 存最后结果
#   4. line 113-131    : get_final_result() / get_token_usage() / reset_for_new_query() — 3 个新方法
#
# byte-identical 部分 (大部分代码) 的详细教学注释见 L0 文件; 本副本只标记 patch 区段。
# 详 [CAESURA LOG_STRUCTURE.md §10.1.2](../../../../../../AllSQPE/CAESURA/LOG_STRUCTURE.md#1012-l1-modify--vendored-包内被改动的文件3-个)。
# =============================================================================
from pathlib import Path
import langchain
import logging
from caesura.model import MyOpenAI
from caesura.phases import PlanningPhase, DiscoveryPhase, MappingPhase, MappingPhase
from langchain.cache import SQLiteCache
from caesura.phases.base_phase import PhaseList
from caesura.phases.runner import RunnerPhase
from caesura.scenarios import get_database
from caesura.tools import ImageSelectTool, SqlTool, TransformTool, VisualQATool, PlottingTool
from caesura.tools.noop import NoopTool
from caesura.tools.text_qa import TextQATool

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
logger = logging.getLogger(__name__)



MAX_NUM_RETRIES = {
    "gpt-3.5-turbo-0613": 3,
    "gpt-4-0613": 1
}

MAX_NUM_ERRORS = {
    "gpt-3.5-turbo-0613": 5,
    "gpt-4-0613": 3
}

class Caesura():
    def __init__(self, database, model_name="gpt-3.5-turbo-0613", interactive=True, log_path=None):
        self.database = database
        self.interactive = interactive
        self.working_memory = dict()
        self.llm = MyOpenAI(temperature=0, model_name=model_name, max_tokens=1024, logging_dir=log_path or ".")
        self.phases = list()
        self.tools = list()
        self.max_num_tries = MAX_NUM_RETRIES[model_name]
        self.max_num_errors = MAX_NUM_ERRORS[model_name]
        self.log_path = log_path
        self.file_handler = None
        # === L1 PATCH (line 41) ===
        # 新增字段, 存最近一次 run(query) 的 final DataFrame (Table 实例).
        # L0 没这个字段 — agent 跑完一个 query 后, 结果就丢了, 无法从外部 *后续访问*.
        # SemBench wrapper 需要 agent.run(q) 后 agent.get_final_result() 拿 DataFrame, 所以加这个 cache.
        self.last_result = None

        # setup
        self.setup_logging()
        self.setup_tools()
        self.setup_phases()

    def setup_logging(self):
        if self.log_path is not None:
            for handler in logging.root.handlers:
                handler.level = logging.root.level
            self.log_path.mkdir(exist_ok=True, parents=True)
            self.file_handler = logging.FileHandler(self.log_path / 'out.log')
            logging.root.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.file_handler.setFormatter(formatter)
            logging.root.addHandler(self.file_handler)

    def setup_tools(self):
        self.tools = list()
        self.tools.append(ImageSelectTool(self.database))
        self.tools.append(VisualQATool(self.database))
        self.tools.append(SqlTool(self.database))
        self.tools.append(TransformTool(self.database, self.llm, self.interactive))
        self.tools.append(PlottingTool(self.database, self.interactive, self.log_path))
        self.tools.append(TextQATool(self.database))
        self.tools.append(NoopTool(self.database))
        for tool in self.tools:
            self.database.register_tool(tool)

    def setup_phases(self):
        self.phases = PhaseList(
            DiscoveryPhase(llm=self.llm, database=self.database, max_num_errors=self.max_num_errors),
            PlanningPhase(llm=self.llm, database=self.database, max_num_errors=self.max_num_errors),
            MappingPhase(llm=self.llm, database=self.database, max_num_errors=self.max_num_errors),
            RunnerPhase(llm=self.llm, database=self.database, max_num_errors=self.max_num_errors),
            reset_on_error=True
        )

    def run(self, query):
        query = query.strip().strip(".")
        error = None
        num_tries = 0
        final_plan = None
        final_result = None
        

        # === L1 PATCH (line 87-88) ===
        # 每次 run(query) 开头 reset MyOpenAI 的 total_*_tokens 计数器为 0.
        # L0 没这个 — token 计数会 *跨 query 累加*, 第二个 query 的 cost 看不出来.
        # MyOpenAI.reset_token_usage 见 L1 patch in model.py.
        # Reset token usage at start of query
        self.llm.reset_token_usage()
        

        while num_tries < self.max_num_tries:
            try:
                final_plan = self.phases.run(query=query, tools=self.tools)
                error = None
                break
            except RuntimeError as e:
                error = self.restart_after_error(e)
            except Exception as e:
                if self.interactive:
                    raise e
                error = self.restart_after_error(e)
            finally:
                num_tries += 1
                final_result = self.database.final_result()
                # === L1 PATCH (line 104) ===
                # 在 clear_working_set() 之前把 final_result 缓存到 self.last_result,
                # 否则下一行 clear_working_set 会让 database.final_result() 返回 None.
                # 这是 L0 → L1 的关键 timing — L0 不需要保留 final_result 跨 clear; L1 需要让 wrapper 后续读取.
                self.last_result = final_result
                self.database.clear_working_set()

        if error is not None:
            logging.root.removeHandler(self.file_handler)
            if self.interactive:
                raise error
            return
        self.log_final_plan(query, final_plan, final_result)
    

    # =========================================================================
    # === L1 PATCH (line 113-131) — 3 个新方法暴露给 SemBench wrapper 用
    # =========================================================================
    # L0 没有这些方法. 它们组成 L1 跨 query 复用 agent 实例的核心 API:
    #   - get_final_result()     : 取上轮 query 的 DataFrame (取 self.last_result, L1 PATCH line 104 填的)
    #   - get_token_usage()      : 转发到 MyOpenAI.get_token_usage(); 拿 prompt/completion/total tokens dict
    #   - reset_for_new_query()  : 把 agent 状态全清 (phases / tools / working_set / token counter), 准备跑下一 query
    #
    # SemBench 调用顺序:
    #   GenericCaesuraRunner.execute_caesura_query()  →
    #     agent.reset_for_new_query()    [清状态]
    #     agent.run(query_text)          [跑 4-Phase]
    #     agent.get_final_result()       [取 DataFrame]
    #   后续 _update_token_usage_and_cost():
    #     agent.get_token_usage()        [拿 token dict]
    # =========================================================================

    def get_final_result(self):
        """Get the final result from the last query execution."""
        # 简单 getter; self.last_result 由 run() 内 L1 PATCH (line 104) 设置
        return self.last_result
    

    def get_token_usage(self):
        """Get token usage from the LLM."""
        # 转发到 MyOpenAI (L1 patched, 见 model.py 新增的 get_token_usage 方法)
        # 返回 dict(prompt_tokens, completion_tokens, total_tokens)
        return self.llm.get_token_usage()
    

    def reset_for_new_query(self):
        """Reset the agent state for a new query execution."""
        # 完整重置 agent 状态 — 让单个 agent 实例能跨 N 个 query 复用而不污染状态。
        # 注意: *不重新创建 MyOpenAI* — 保留 LLM 实例 (省去模型重加载和 langchain SQLiteCache 初始化)
        # 与 restart_after_error 的区别: 后者会 *new 一个 MyOpenAI* 并提高 temperature, 这里只 reset 状态
        # Reset phases
        self.setup_phases()
        # Reset tools 
        self.setup_tools()
        # Clear database working set
        self.database.clear_working_set()
        # Reset token usage
        self.llm.reset_token_usage()

    def restart_after_error(self, e):
        logger.warning(e, exc_info=True)
        self.setup_tools()
        self.setup_phases()
        error = e
        self.llm = MyOpenAI(temperature=self.llm.temperature + 0.2,
                            model_name=self.llm.model_name, max_tokens=1024)
        return error

    def log_final_plan(self, query, final_plan, final_result):
        plan_str = final_plan.final_format(query)
        print()
        print(plan_str)
        result_str = final_result.data_frame.to_markdown() if final_result is not None else None
        print()
        print(result_str)

        if self.log_path is not None:
            path = Path(self.log_path) / "final-plan.log" 
            with open(path, "w") as f: 
                print(plan_str, file=f)

            if result_str is not None:
                path = Path(self.log_path) / "final-result.log"
                with open(path, "w") as f: 
                    print(result_str, file=f)
        logging.root.removeHandler(self.file_handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


    model = {"3": "gpt-3.5-turbo-0613", "4": "gpt-4-0613"}[input("Model (GPT-3/GPT-4): GPT-").strip()]
    dataset_name = input("Dataset (artwork/rotowire): ").strip()
    dl = get_database(dataset_name, sampled=False)
    agent = Caesura(dl, model_name=model, interactive=False)
    agent.run(input("Query : ").strip())

    # agent.run("For every player, what is the highest number of points they scored in a game?")
    # agent.run("Plot the number religious artworks from each century.")
    # agent.run("Plot the average number of persons depicted in the paintings of each genre.")
    # agent.run("Plot the number of paintings that depict Madonna and Child for each century.")

    # fake_llm_responses=[]
