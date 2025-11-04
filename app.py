import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Occupancy Cluster Checker", layout="wide")

# ---------------------------
# 配置文件名
# ---------------------------
TRUTH_FILE = Path("Raw_Occ_cluster_Truth.xlsx")   # 老师的真值
USER_FILE = Path("users.xlsx")                    # 学号+密码+最好成绩

st.title("📈 Occupancy Cluster Checker")

# ---------------------------
# 读标准答案
# ---------------------------
if not TRUTH_FILE.exists():
    st.error("❗ 没找到标准答案文件 `Raw_Occ_cluster_Truth.xlsx`，请放到 app.py 同目录下。")
    st.stop()
df_truth = pd.read_excel(TRUTH_FILE, index_col=0)
if df_truth.shape[1] != 3:
    st.warning(f"标准答案文件列数是 {df_truth.shape[1]}，不是 3 列，请检查老师文件。")

# ---------------------------
# 读用户表
# ---------------------------
if not USER_FILE.exists():
    st.error("❗ 没找到用户文件 `users.xlsx`，请先创建一个包含 student_id, password, best_dist 三列的Excel。")
    st.stop()

def load_users() -> pd.DataFrame:
    dfu = pd.read_excel(USER_FILE)
    # 保证列存在
    if "student_id" not in dfu.columns or "password" not in dfu.columns:
        st.error("`users.xlsx` 中必须包含列: student_id, password")
        st.stop()
    if "best_dist" not in dfu.columns:
        dfu["best_dist"] = np.nan
    return dfu

def save_users(dfu: pd.DataFrame):
    dfu.to_excel(USER_FILE, index=False)

users_df = load_users()

# ---------------------------
# 工具函数
# ---------------------------
def read_uploaded_excel(file):
    suffix = Path(file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file, index_col=0)
    elif suffix == ".csv":
        return pd.read_csv(file, index_col=0)
    else:
        raise ValueError("只支持 .xlsx / .xls / .csv")

def sort_cols_by_mean(df: pd.DataFrame) -> pd.DataFrame:
    col_means = df.mean(axis=0)
    sorted_cols = col_means.sort_values().index.tolist()
    return df[sorted_cols]

def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# ---------------------------
# 登录区域
# ---------------------------
col1, col2, col3 = st.columns([2, 4, 2])
with col2:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if not st.session_state.logged_in:
        st.subheader("🔐 登录")
        sid = st.text_input("学号", value="", key="login_sid")
        pwd = st.text_input("密码", value="", type="password", key="login_pwd")
        if st.button("登录"):
            # 检查账号密码
            match = users_df[
                (users_df["student_id"].astype(str) == str(sid).strip()) &
                (users_df["password"].astype(str) == str(pwd).strip())
            ]
            if len(match) == 1:
                st.session_state.logged_in = True
                st.session_state.current_user = str(sid).strip()
                st.success(f"登录成功，欢迎 {sid} ！")
            else:
                st.error("学号或密码错误，请重试。")
        st.stop()  # 没登录就不往下走

# ---------------------------
# 登录后界面
# ---------------------------
col1, col2, col3 = st.columns([2, 4, 2])
with col2:
    st.info(f"当前登录：{st.session_state.current_user}")

    uploaded = st.file_uploader(
        "请上传你的聚类代表曲线（Excel共3列，第一列为时间/序号，后面3列为聚类代表曲线）",
        type=["xlsx", "xls", "csv"]
    )

if uploaded is not None:
    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:
        # 读取学生文件
        try:
            df_stu = read_uploaded_excel(uploaded)
        except Exception as e:
            st.error(f"读取学生文件失败：{e}")
            st.stop()

        # 检查列数
        if df_stu.shape[1] != 3:
            st.error(f"文件为 {df_stu.shape[1]} 列，应为 3 列。请检查输入表格格式。")
            st.stop()

        # 排序对齐
        df_truth_sorted = sort_cols_by_mean(df_truth)
        df_stu_sorted = sort_cols_by_mean(df_stu)

        # 行对齐
        if not df_truth_sorted.index.equals(df_stu_sorted.index):
            df_stu_sorted = df_stu_sorted.reindex(df_truth_sorted.index)

        # 计算三个欧氏距离
        dists = []
        for i in range(3):
            col_truth = df_truth_sorted.iloc[:, i].values
            col_stu = df_stu_sorted.iloc[:, i].values
            d = euclidean_dist(col_truth, col_stu)
            dists.append(d)
        sum_dist = float(np.sum(dists))

        st.subheader("📏 本次结果")
        st.success(f"👉 本次欧氏距离之和：**{sum_dist:.4f}**")

        # ====== 更新用户最高分（最小dist） ======
        sid = st.session_state.current_user
        users_df = load_users()  # 再读一遍，防止多人同时操作时被覆盖
        idx = users_df[users_df["student_id"].astype(str) == sid].index
        if len(idx) == 1:
            old_best = users_df.loc[idx[0], "best_dist"]
            # 如果还没成绩 or 这次更好，就更新
            if pd.isna(old_best) or sum_dist < old_best:
                users_df.loc[idx[0], "best_dist"] = sum_dist
                save_users(users_df)
                st.success("🎉 恭喜！你创造了自己的最好成绩，系统已记录。")
            else:
                st.info(f"你之前的最好成绩是 {old_best:.4f}，本次没有更新。")
        else:
            st.error("当前登录用户在 users.xlsx 中不存在，请检查账号。")

    # ====== 画图 ======
    st.subheader("📊 曲线对比")
    time_axis = df_truth_sorted.index
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=140)

    tick_positions = [4, 20, 36, 52, 68, 84]
    tick_labels = ["1:00", "5:00", "9:00", "13:00", "17:00", "21:00"]

    for i in range(3):
        ax = axes[i]
        ax.plot(time_axis, df_truth_sorted.iloc[:, i].values, label="Truth", linewidth=2)
        ax.plot(time_axis, df_stu_sorted.iloc[:, i].values, label="Student", linewidth=1.5, linestyle="--")
        ax.set_title(f"Cluster {i} (Dist={dists[i]:.3f})")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------
# 排行榜
# ---------------------------
# ✅ 居中显示排行榜（[2,4,2]布局）
col1, col2, col3 = st.columns([2, 4, 2])
with col2:
    st.subheader("🏆 排行榜")
    users_df = load_users()

    # 只显示有成绩的
    rank_df = users_df.dropna(subset=["best_dist"]).copy()
    rank_df = rank_df.sort_values("best_dist", ascending=True).reset_index(drop=True)
    rank_df.index = rank_df.index + 1  # ✅ 排名从1开始
    rank_df = rank_df.rename(columns={"student_id": "学号", "best_dist": "得分"})

    # ✅ 转换为HTML并自定义样式
    html_table = rank_df[["学号", "得分"]].to_html(classes="styled-table", justify="center", border=0)

    st.markdown(
        """
        <style>
        .styled-table {
            font-size: 22px;                /* ✅ 字体更大 */
            text-align: center;             /* ✅ 内容居中 */
            margin: 0 auto;                 /* ✅ 表格居中 */
            border-collapse: collapse;
            width: 60%;                     /* ✅ 表格宽度更大（占中栏的90%） */
        }
        .styled-table th {
            background-color: #f2f2f2;      /* ✅ 表头浅灰背景 */
            font-weight: bold;
            font-size: 24px;                /* ✅ 表头更大 */
            padding: 5px 8px;             /* ✅ 表头留白略增 */
        }
        .styled-table td {
            padding: 5px 8px;             /* ✅ 单元格留白略增 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ✅ 输出排行榜HTML
    st.markdown(html_table, unsafe_allow_html=True)
