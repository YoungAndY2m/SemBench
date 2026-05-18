# =============================================================================
# SemBench L1 vendored 副本 — caesura/scenarios.py
# =============================================================================
# 教学注释 pass (L1 MODIFY) by Claude.
# 这是 L0 [AllSQPE/CAESURA/caesura/scenarios.py](../../../../../../AllSQPE/CAESURA/caesura/scenarios.py) 的 vendored 副本,
# 含 3 处 patch (相对 L0 +23 行), 加 movie scenario 注册。
#
# Patch 位置 (本文件):
#   1. line 2-3        : import os                              — 未使用 (dead import)
#   2. line 11-12      : if scenario == "movie": ...            — get_database dispatcher 加 movie 分支
#   3. line 54-72      : def movie_scenario(sampled=True): ...  — 新增 movie scenario 工厂函数
#
# ⚠ 关键 finding (LOG_STRUCTURE.md §10.1.2 C 表底部):
# `movie_scenario()` 实际 *不被运行时调用* — 因为 `GenericCaesuraRunner._setup_database_from_files`
# (generic_caesura_runner.py:83-117) 用 `self.data_path` 走自己的 path; SemBench 的 movie 数据
# 在 files/movie/data/sf_2000/, 而非 datasets/movie/ (本文件 hardcode 的路径).
# 即 movie_scenario() 是 *dead code* — vendored 期 patched 但实际生产路径绕开。
# 详 [LOG.md "遗留问题 #5"](../../../../../../AllSQPE/CAESURA/LOG.md)
#
# byte-identical 部分 (artwork_scenario / rotowire_scenario) 详 L0 教学注释。
# =============================================================================
from caesura.database.database import Database
import datetime
# === L1 PATCH (line 3) — dead import (未使用)
# 推测早期版本想用 os.path.join 处理跨平台路径; 实际本文件用 hardcoded "datasets/movie/..." 字符串
import os


def get_database(scenario, sampled=True):
    if scenario == "artwork":
        return artwork_scenario(sampled=sampled)
    if scenario == "rotowire":
        return rotowire_scenario(sampled=sampled)
    # === L1 PATCH (line 11-12) ===
    # 加 movie 分支让 get_database("movie") 不会 KeyError; 但 movie_scenario() 实际 *没人调*,
    # 因为 GenericCaesuraRunner 走自己的 _setup_database_from_files (绕开 get_database).
    # 见 file docstring "关键 finding" 解释
    if scenario == "movie":
        return movie_scenario(sampled=sampled)
    raise KeyError(scenario)


def artwork_scenario(sampled=True):
    dl = Database()
    dl.add_tabular_table("paintings_metadata", f"datasets/art/paintings{'_sampled' if sampled else ''}.csv",
                           "a table that contains general information about paintings", path_columns=("img_path",))
    mask = dl._tables["paintings_metadata"].data_frame["inception"].apply(lambda x: not x.startswith("http"))
    dl._tables["paintings_metadata"].data_frame = dl._tables["paintings_metadata"].data_frame[mask].reset_index(drop=True)
    dl.add_image_table("painting_images", "datasets/art/images",
                         "a table that contains images of paintings",
                         file_paths=dl.get_column_values("paintings_metadata", "img_path").tolist())
    dl.link_image("paintings_metadata", "painting_images", "img_path")
    dl.build_relevant_values_index("paintings_metadata", "genre", "movement")
    return dl


def rotowire_scenario(sampled=True):
    dl = Database()
    dl.add_tabular_table("players", "datasets/rotowire/players.csv",
                         "a table that contains general information about basketball players")
    dl.add_tabular_table("teams", "datasets/rotowire/teams.csv",
                         "a table that contains general information about basketball teams")
    dl.add_tabular_table("players_to_games", "datasets/rotowire/players_to_games.csv",
                         "a table that maps players to games")
    dl.add_tabular_table("teams_to_games", "datasets/rotowire/teams_to_games.csv",
                         "a table that maps teams to games")
    dl.add_text_table("game_reports", f"datasets/rotowire/reports{'_sampled' if sampled else ''}.csv",
                      "a table containing game reports about basketball games. Each report contains statistics about all the teams and players that participated in a single game, e.g. number of points scored by each player / team, number of assists by each player / team, etc.")
    dl.link("players_to_games", "game_reports", "game_id")
    dl.link("teams_to_games", "game_reports", "game_id")
    dl.link("teams", "teams_to_games", "name")
    dl.link("players", "players_to_games", "name")
    dl.build_relevant_values_index("players", "name", "nationality", "position")
    dl.build_relevant_values_index("teams", "arena", "location", "president", "coach")
    dl.tables["players"].data_frame["birth_date"].apply(
        lambda x: datetime.datetime.strptime(x, "%d.%m.%Y").strftime("%Y-%m-%d")
    )
    return dl


# =============================================================================
# === L1 PATCH (line 54+) — 新增 movie scenario 工厂
# =============================================================================
# ⚠ 注意! 本函数 *实际不被运行时调用* — 详 file docstring "关键 finding":
#   - hardcoded path "datasets/movie/movies.csv" 与 SemBench 实际 path "files/movie/data/sf_2000/Movies.csv" 不符
#   - 真实生产路径走 GenericCaesuraRunner._setup_database_from_files() 而非这里
# 保留是为了兼容 `python caesura/main.py` CLI 入口的 get_database 调用; 但 CLI 路径没人用了。
#
# 参数 sampled=True: 与其它 scenario 一致的 API; 本函数也忽略了它 (没生效)
# =============================================================================
def movie_scenario(sampled=True):
    dl = Database()
    

    # Add movies metadata table
    # hardcoded "datasets/movie/movies.csv" — 相对 cwd; CAESURA CLI 工作目录里没这文件
    dl.add_tabular_table("movies", "datasets/movie/movies.csv",
                         "a table that contains information about movies including titles, scores, ratings, genres, directors, and other metadata")
    

    # Add reviews table with reviewText as last column (for TEXT datatype)
    # add_text_table 约定: 最后一列被标 TEXT — 假设 reviews.csv 末列是 reviewText
    dl.add_text_table("reviews", "datasets/movie/reviews.csv",
                      "a table that contains movie reviews with metadata and review text content for sentiment analysis")
    

    # Link reviews to movies by movie id
    # 单向 link: reviews.id ↔ movies.id (与 generic_caesura_runner.py 内的 link 一致)
    dl.link("reviews", "movies", "id")
    

    # Build relevant values index for key categorical columns
    # 注意! reviewState 列在 movie/caesura_runner.py:_setup_database_from_files 里被 drop 了;
    # 这里却给它建索引 — 与 per-scenario subclass 的清洗逻辑不一致 (反正本函数不被调, 无后果)
    dl.build_relevant_values_index("movies", "genre", "director", "rating", "originalLanguage")
    dl.build_relevant_values_index("reviews", "reviewState", "publicationName", "isTopCritic")
    

    return dl
